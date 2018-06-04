"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
using shared arrays from the initial module binded with maximum shape.
"""

"""
this file is derived from one of the official examples, additional features:
1. {acc_backward},  {acc_update}
    for larger batch_size since in many cases where there is a need of this file the batch dim is always is ONE
2. {save_checkpoint}
    if shape has veried, barely using self._curr_module.save_checkpoint could book you a nightmare in some day, use this!

TODO:
1. interface for load like *mx.mod.Module.load*

CHANGELOG:
    1. provides acc_fit taking mod as arg --Oct 22, 2017

Chen Y. Liang
"""




import logging
import time
from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module
import mxnet as mx
metric = mx.metric
BatchEndParam = mx.model.BatchEndParam
_as_list = mx.base._as_list

def acc_grad_arrays(mod, arg_acc_grad_arrays=None):
    """
        if grads is None, return the copied one else, return accumulated one
    """
    if arg_acc_grad_arrays is None:
        return [ [grad.copy() if grad is not None else None for grad in grads]\
                    for grads in mod._exec_group.grad_arrays  ]
    else:
        # accumulate...
        return [ [grad + acc_grad if grad is not None else None for\
                                    grad, acc_grad in zip(grads, acc_grads) ]\
                            for grads, acc_grads in zip(mod._exec_group.grad_arrays, arg_acc_grad_arrays) ]

def set_grad_arrays(mod, arg_acc_grad_arrays, normsize=1):
    for acc_grads, mod_grads in zip(arg_acc_grad_arrays, mod._exec_group.grad_arrays):
        for acc_grad, mod_grad in zip(acc_grads, mod_grads):
            if acc_grad is not None:
                mod_grad[:] = acc_grad[:]/normsize
#    mod._exec_group.grad_arrays=[[grad.copyto(grad.context)/normsize if grad is not None else None\
#                    for grad in grads] for grads in arg_acc_grad_arrays]


def acc_fit(mod, update_batch_size,\
            train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        """
    this function aims to support training in larger input size by
                allocating additive space to store auxiliary grads
        mod:
            mx.mod.Module
        update_batch:
            int, specifying how many batches between two updates
        **arg_keys:
            same as mod.fit
        """
        assert num_epoch is not None, 'please specify number of epochs'
        it_batch_size=train_data.batch_size

        mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            mod.install_monitor(monitor)
        mod.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        mod.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        arg_acc_grad_arrays=None # to store auxiliary grad_arrays
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                mod.forward_backward(data_batch)
                arg_acc_grad_arrays = acc_grad_arrays(mod, arg_acc_grad_arrays)
                if nbatch*it_batch_size % update_batch_size==0 and nbatch >0:
                    set_grad_arrays(mod, arg_acc_grad_arrays, update_batch_size/it_batch_size) # normsize=1 by default(softmax norm)
                    mod.update()
                    arg_acc_grad_arrays=None
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    mod.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True

                mod.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1
            if arg_acc_grad_arrays is not None:# left one update...
                set_grad_arrays(mod, arg_acc_grad_arrays, update_batch_size/it_batch_size)
                mod.update()
                arg_acc_grad_arrays = None

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                mod.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            mod.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = mod.get_params()
            mod.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, mod.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = mod.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    mod.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()






class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """
    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._symbol = symbol
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix

        self.grad   =None   #  use this one for acc !


        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()
#            assert 0

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes))

        max_data_shapes = list()
        for name, shape in data_shapes:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if label_shapes is not None:
            for name, shape in label_shapes:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None

        module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        module.bind(max_data_shapes, max_label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)

#	assert 0

        self._curr_module = module


#	print(self._curr_module.optimizer_initialized)

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)
  #          print(arg_params)
 #           assert 0
    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = dict(self._curr_module.data_shapes + self._curr_module.label_shapes)
        else:
            current_shapes = dict(self._curr_module.data_shapes)

        # get input_shapes
        if data_batch.provide_label is not None:
            input_shapes = dict(data_batch.provide_data + data_batch.provide_label)
        else:
            input_shapes = dict(data_batch.provide_data)

        # decide if shape changed
        shape_changed = False
        for k, v in current_shapes.items():
            if v != input_shapes[k]:
                shape_changed = True

        if shape_changed:
            module = Module(self._symbol, self._data_names, self._label_names,
                            logger=self.logger, context=self._context,
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names)
            module.bind(data_batch.provide_data, data_batch.provide_label, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

        self._curr_module.forward(data_batch, is_train=is_train)

    def __acc_grad(self,mod,in_grads):
        if in_grads is None:
#            print('grad is None')
            
            ret_grads= [[grad.copyto(grad.context) if grad is not None else None for grad in grads] for grads in mod._exec_group.grad_arrays]
        else:
            ret_grads=[[grad1.copyto(grad1.context) + grad0 if grad1 is not None else None  for grad1,grad0 in zip(grads1,grads0)]\
                        for grads1,grads0 in zip(mod._exec_group.grad_arrays,in_grads)]
        return ret_grads

    def acc_backward(self, out_grads=None):
        assert self.binded and self.params_initialized
 

        self._curr_module.backward(out_grads=out_grads)
        self.grad=self.__acc_grad(self._curr_module,self.grad)


    def acc_update(self,normsize=1):
        assert self.binded and self.params_initialized and self.optimizer_initialized
#        self._curr_module._exec_group.grad_arrays=None
#        self._curr_module._exec_group.grad_arrays=\
#                      [[grad.copyto(grad.context)*1 if grad is not None else None for grad in grads] for grads in self.grad]

        for acc_grads, mod_grads in zip(self.grad,self._curr_module._exec_group.grad_arrays):
            for acc_grad, mod_grad in zip(acc_grads, mod_grads):
                if acc_grad is not None:
                    mod_grad[:] = acc_grad[:]/normsize

#        try:
#            del(self._curr_module._exec_group.execs[0].grad_arrays)
#        except:
#            None
        self._curr_module.update()
        self.grad = None


    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)

    def save_checkpoint(self,prefix, epoch, save_optimizer_states=False):
        self._curr_module._sync_params_from_devices()
        self._curr_module.save_checkpoint(prefix, epoch, save_optimizer_states)



