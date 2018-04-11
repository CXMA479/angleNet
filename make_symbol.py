import mxnet as mx
from config import config as cfg
import os, sys, logging
from module.module import MutableModule
import numpy as np


def gen_symbol():
    """
                 generate full symbol, fixed parameter names are also returned
    """
    target_label = mx.sym.Variable('target_label')
    gdt_bbox         = mx.sym.Variable('target_bbox')
    rpn_outside_weight = mx.sym.Variable('rpn_outside_weight')
    rpn_inside_weight    = mx.sym.Variable('rpn_inside_weight')


    symbol = mx.sym.load('%s-symbol.json'%cfg.net.symbol_path)

    ################################
    ### construct feature maps...###
    ################################
    feat_sym_1x1_0 = symbol.get_internals()[cfg.net.rpn_conv_names['1x1'][0]+'_output']


#    print(feat_sym_2x2_0.list_arguments())
    fixed_param_names = []
    for _ in [ feat_sym_1x1_0,]:# feat_sym_2x2_0]: # fixed_param_names will be returned to the caller
        fixed_param_names += _.list_arguments()
#    print(set(fixed_param_names))
    fixed_param_names = list(set(fixed_param_names)) # uniquify...

    feat_sym_1x1_0 = mx.sym.Convolution(feat_sym_1x1_0, kernel=(3,3),pad=(1,1), num_filter=256, no_bias=True)
    feat_sym_1x1_0 = mx.sym.Activation(feat_sym_1x1_0, act_type='relu')
    feat_sym       = feat_sym_1x1_0
    feat_sym_2x2 = None
    if '4x4' in cfg.net.rpn_conv_names:
      feat_sym_4x4_0 = symbol.get_internals()[cfg.net.rpn_conv_names['4x4'][0]+'_output']
      feat_sym_4x4_0 = mx.sym.BlockGrad(feat_sym_4x4_0)
      feat_sym_2x2 = mx.sym.Convolution(feat_sym_4x4_0, kernel=(2,2), stride=(2,2),\
                                            num_filter=256, no_bias=True)
      feat_sym_2x2 = mx.sym.Activation(feat_sym_2x2,act_type='relu')    

          
    if '2x2' in cfg.net.rpn_conv_names: 
      feat_sym_2x2_0 = symbol.get_internals()[cfg.net.rpn_conv_names['2x2'][0]+'_output']
      feat_sym_2x2_0 = mx.sym.BlockGrad(feat_sym_2x2_0)
      feat_sym_2x2   = mx.sym.concat(feat_sym_2x2, feat_sym_2x2_0) \
                        if feat_sym_2x2 is not None else feat_sym_2x2_0
      feat_sym_2x2 = mx.sym.Convolution(feat_sym_2x2, kernel=(2,2), stride=(2,2),\
                                            num_filter=256, no_bias=True)
      feat_sym_2x2 = mx.sym.Activation(feat_sym_2x2, act_type='relu')
      
    feat_sym = mx.sym.concat(feat_sym_1x1_0, feat_sym_2x2) if feat_sym_2x2 is not None else feat_sym_1x1_0


#    feat_sym_2x2_0 = mx.sym.Pooling(feat_sym_2x2_0, kernel=(2,2), stride=(2,2), pool_type='avg', pooling_convention='full')
#    feat_sym_2x2_0 = mx.sym.Convolution(feat_sym_2x2_0, kernel=(3,3), pad=(1,1), num_filter=256)
#    feat_sym_2x2_0 = mx.sym.Activation(feat_sym_2x2_0, act_type='tanh')
    #[feat_sym_1x1_0, feat_sym_2x2_0]= [mx.sym.BatchNorm(_) for _ in [feat_sym_1x1_0, feat_sym_2x2_0] ]

   
    
#    feat_sym_2x2_0 = mx.sym.Convolution(feat_sym_2x2_0, kernel=(3,3), pad=(1,1), num_filter=256)
#    feat_sym_2x2_0 = mx.sym.Convolution(feat_sym_2x2_0, kernel=(3,3), pad=(1,1), num_filter=256)

    # concatenate...

#    in_shape,out_shape_2x2 ,uax_shape=feat_sym_2x2_0.infer_shape( data=(1,3, 448, 448))
    in_shape,out_shape_1x1 ,uax_shape=feat_sym_1x1_0.infer_shape(data=(1,3, 448, 448))
#    print('1x1: '+str(out_shape_1x1),'2x2: '+str(out_shape_2x2) )
#    assert 0


    
    #feat_sym = mx.sym.Convolution(feat_sym, kernel=(3,3), pad=(1,1),num_filter=1024/2,\
    #                            no_bias=True)
    #feat_sym = mx.sym.BatchNorm(feat_sym)
    #feat_sym = mx.sym.Activation(feat_sym, act_type='tanh')
    """
    feat_sym = mx.sym.Convolution(feat_sym, kernel=(3,3), pad=(1,1), num_filter=1024,\
                                no_bias=True)
    feat_sym = mx.sym.BatchNorm(feat_sym)
    feat_sym = mx.sym.Activation(feat_sym, act_type='relu')
    """
    ###########################
    ###   feature maps over ###
    ###########################
    cls_symbol  = feat_sym
    reg_symbol  = feat_sym
    # symbol is shared by both bbox & score...
    """
    reg_symbol = mx.sym.Convolution(reg_symbol, kernel=(1,1), num_filter=1024,name='rpn_conv1',\
                                no_bias=True)
    reg_symbol = mx.sym.BatchNorm(reg_symbol)
    reg_symbol = mx.sym.Activation(reg_symbol, act_type='relu',name='rpn_relu')
    """



#    symbol = mx.sym.LeakyReLU(symbol, act_type='leaky',name='rpn_leaky')  # 1 x 512 x H x W
#    symbol = mx.sym.Activation(symbol, act_type='relu',name='rpn_relu')  # 1 x 512 x H x W

    """
    symbol = mx.sym.Convolution(symbol, kernel=(3,3), pad=(1,1), num_filter=512*2,name='rpn_conv2',\
                                no_bias=True)
    symbol = mx.sym.BatchNorm(symbol)
    symbol = mx.sym.Activation(symbol, act_type='relu',name='rpn_relu')
    """

    """
        for target label prediction.
    """

#    x = symbol
#    x = mx.sym.Convolution(x, kernel=(3,3), pad=(1,1), num_filter=512,name='rpn_conv_softmax1')
#    x = mx.sym.Activation(x, act_type='relu',name='rpn_relu_sftmx2')  # 1 x 512 x H x W
#    symbol = x
    """
    logging.info('Block the grad of the score for debug...')
    X = symbol
    X = mx.sym.BlockGrad(X)
    symbol = X
    """
    """
    logging.info('#'*10+'Block Grad for symbol'+'#'*10)
    x = symbol#rpn_bbox_loss_
    x = mx.sym.BlockGrad(x)
    symbol = x
    """
    """
    cls_symbol = mx.sym.Convolution(cls_symbol, kernel=(1,1), num_filter=1024,name='cls_conv1',\
                                no_bias=True)
    cls_symbol = mx.sym.BatchNorm(cls_symbol)
    cls_symbol = mx.sym.Activation(cls_symbol, act_type='relu',name='cls_relu')
    """
    symbol_score = mx.sym.Convolution(cls_symbol ,kernel=(1,1), num_filter=2*cfg.type_num, name='rpn_score')
    #symbol_score = mx.sym.BatchNorm(symbol_score)
#    symbol_score = mx.sym.Activation(symbol_score, act_type='relu')
    symbol_score = mx.sym.transpose(symbol_score,axes=(0,2,3,1))    # channel must be at the last
    reshape_symbol_score = mx.sym.reshape(symbol_score, (0,-1,2 ),name='rpn_score_reshape' ) # num x HW x 2
    # preserve_shape=True: softmax on the last dim
    softmax_output = mx.sym.SoftmaxOutput(reshape_symbol_score, label=target_label, preserve_shape=True,\
                        use_ignore=True, ignore_label=-1, name='rpn_label_loss')

#    score_mask = mx.sym.argmax(softmax_output, axis=-1, keepdims=True) # (b, HW, 1)

    pred_bbox = mx.sym.Convolution(reg_symbol, kernel=(1,1), num_filter=5*cfg.type_num, name='rpn_pred_bbox')
    pred_bbox = mx.sym.transpose(pred_bbox,axes=(0,2,3,1))    # see lab/hist/it/anchor_vs_pred.py for the details
    reshape_pred_bbox_ = mx.sym.reshape(pred_bbox, (0,-1,5),name='rpn_pred_bbox_reshape' )

#    logging.info('Block the grad of the score for debug...')
#    x = reshape_pred_bbox
#    x = mx.sym.BlockGrad(x)
#    reshape_pred_bbox = x


    rpn_bbox_loss_ =     rpn_outside_weight * mx.sym.smooth_l1(\
               data= rpn_inside_weight* (reshape_pred_bbox_ - gdt_bbox) ,\
                 scalar=cfg.train.l1_smooth_sclr,    name='rpn_bbox_loss_l1_smooth')


#    rpn_bbox_loss_ = rpn_outside_weight * rpn_inside_weight* mx.sym.abs(reshape_pred_bbox - gdt_bbox)

    rpn_pred_bbox = mx.sym.BlockGrad(reshape_pred_bbox_)
    
    
    rpn_bbox_loss = mx.sym.MakeLoss(rpn_bbox_loss_*cfg.train.bbox_scalar,name='rpn_bbox_loss')/cfg.train.bbox_scalar


    symbol = mx.sym.Group([ softmax_output,  rpn_pred_bbox, rpn_bbox_loss])#, mx.sym.BlockGrad(symbol) ] )


    return symbol, fixed_param_names




def batch_set_mult_lr(mod, relaxed_param_names, lr):
    """
        set all params in lr learning-rate
    """
    logging.info('relaxed params: %f'%lr)
    logging.info(relaxed_param_names)
    name_lr_dict = {param_name:lr for param_name in relaxed_param_names}
    mod._optimizer.set_lr_mult(name_lr_dict)




def gen_model(symbol_model,fixed_param_names,it, data_names=cfg.it.dataNames,\
        label_names=cfg.it.labelNames,softfixed_param_names=None, softfixed_mul_lr=0 ):
    """
              generate and initilize the model
                            use MutableModule
    """
    if fixed_param_names is None:
        assert softfixed_param_names is not None
    if softfixed_param_names is None:
        assert fixed_param_names is not None
    mod = MutableModule(symbol_model, data_names, label_names, context=cfg.train.ctx,fixed_param_prefix=fixed_param_names)
    mod.bind(it.provide_data, label_shapes=it.provide_label)
    _, arg_params, aux_params = mx.model.load_checkpoint(cfg.net.symbol_path,cfg.net.params_epoch)
    mod.init_params(mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params,allow_missing=True)
    mod.init_optimizer(optimizer_params=(  ('learning_rate',cfg.train.lr),('wd',cfg.train.wd),('momentum',cfg.train.momentum),('clip_gradient',cfg.train.clip_gradient)  ) )
    if softfixed_param_names is not None:
        batch_set_mult_lr(mod._curr_module,softfixed_param_names, softfixed_mul_lr)
    return mod

def feval_l1_angleMetric( label, pred):
    """    just abs_sum the pred """
#    logging.info( np.abs(pred).sum() )
#    logging.info(pred.size)
    return np.abs(pred).sum()*1./(label>0).sum()

def feval_acc_angleMetric(label, pred):
#    assert 0, (pred.shape, label.shape)
    pred = np.argmax(pred, axis=2)
    pred, label = [ x.astype('int32') for x in [pred, label] ]
    # return sum_metric, num_inst
#    logging.debug('whole cls samples:%d\tpostive samples:%d\tnegative samples:%d'%((label>=0).sum(),(label>0).sum(), (label==0).sum()) )
    sum_metric, num_inst  = np.sum( pred[label>=0].flat == label[label>=0].flat ), (label>=0).sum()
    return  sum_metric, num_inst





class AngleMetric(mx.metric.CustomMetric):
    """
        supports passing control_idx into feval
    """
    def __init__ (self, feval, **arg_keys):
        super(AngleMetric).__init__(feval, **arg_keys)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
                The labels of the data.

        preds : list of `NDArray`
                Predicted values.
        """
        if not self._allow_extra_outputs:
            check_label_shapes(labels, preds)

        for idx,(pred, label) in enumerate( zip(preds, labels) ):
            label = label.asnumpy()
            pred = pred.asnumpy()

            reval = self._feval(idx, label, pred)
            if isinstance(reval, tuple):
                (sum_metric, num_inst) = reval
                self.sum_metric += sum_metric
                self.num_inst += num_inst
            else:
                self.sum_metric += reval
                self.num_inst += 1






