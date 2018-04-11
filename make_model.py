"""
to use interactive iteration, we need gloun, to use gluon we need re-write make_symbol.

gen_feat_model, gen_cfd_model, gen_bbx_model      # cfd: confidence

Chen Y. Liang
"""
import mxnet as mx
from config import config as cfg
import os, sys, logging
from module.module import MutableModule
import numpy as np
from mxnet import gluon

def batch_set_mult_lr(optimizer, relaxed_param_names, lr):
    """
        set all params in lr learning-rate
    """
    logging.info('relaxed params: %f'%lr)
    logging.info(relaxed_param_names)
    name_lr_dict = {param_name:lr for param_name in relaxed_param_names}
    optimizer.set_lr_mult(name_lr_dict)



def gen_feat_model():
    # this model calls gluon.nn.SymbolBlock for pakaging feature symbol.
    symbol = mx.sym.load('%s-symbol.json'%cfg.net.symbol_path)
#    symbol, arg_params, aux_params = mx.model.load_checkpoint(cfg.net.symbol_path, cfg.net.epoch)

    ################################
    ### construct feature maps...###
    ################################
    feat_sym_1x1_0 = symbol.get_internals()[cfg.net.rpn_conv_names['1x1'][0]+'_output']

    preloaded_param_names = []
    for _ in [ feat_sym_1x1_0,]:# feat_sym_2x2_0]: # fixed_param_names will be returned to the caller
        preloaded_param_names += _.list_arguments()
#    fixed_param_names = list(set(fixed_param_names)) # uniquify...

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
    model = gluon.nn.SymbolBlock(feat_sym, feat_sym.get_internals()['data'])
    model.load_params('%s-%04d.params'%(cfg.net.symbol_path, cfg.net.params_epoch), cfg.train.ctx,allow_missing=True, ignore_extra=True)
    # Xavier cannot init bais
    # so this line checks previous line
    model.initialize(mx.init.Xavier(), ctx=cfg.train.ctx)
    #
    # build trainer...
    ## Optimizer
    optimizer = mx.optimizer.SGD(cfg.train.momentum, clip_gradient=cfg.train.clip_gradient)
    optimizer.set_learning_rate(cfg.train.lr)
    ##  set lr_mult... if any
    if cfg.train.mult_lr is not None:
        batch_set_mult_lr(optimizer, preloaded_param_names, cfg.train.mult_lr)
    Trainer = gluon.Trainer(model.collect_params(), optimizer)
    model.hybridize()
    return model, Trainer
    
        #    TODO: 
def gen_cfd_model():
    symbol_score = mx.sym.Convolution(cls_symbol ,kernel=(1,1), num_filter=2*cfg.type_num, name='rpn_score')
    #symbol_score = mx.sym.BatchNorm(symbol_score)
#    symbol_score = mx.sym.Activation(symbol_score, act_type='relu')
    symbol_score = mx.sym.transpose(symbol_score,axes=(0,2,3,1))    # channel must be at the last
    reshape_symbol_score = mx.sym.reshape(symbol_score, (0,-1,2 ),name='rpn_score_reshape' ) # num x HW x 2
    # preserve_shape=True: softmax on the last dim
    softmax_output = mx.sym.SoftmaxOutput(reshape_symbol_score, label=target_label, preserve_shape=True,\
                        use_ignore=True, ignore_label=-1, name='rpn_label_loss')






def gen_bbx_model():
    pred_bbox = mx.sym.Convolution(reg_symbol, kernel=(1,1),\
                    num_filter=5*cfg.type_num, name='rpn_pred_bbox')
    pred_bbox = mx.sym.transpose(pred_bbox,axes=(0,2,3,1))    # see lab/hist/it/anchor_vs_pred.py for the details
    reshape_pred_bbox_ = mx.sym.reshape(pred_bbox, (0,-1,5),name='rpn_pred_bbox_reshape' )

class BBxLosser():
    # bounding-box loss measurement
    def __init__(self,sclr=cfg.train.l1_smooth_sclr, bbx_loss_scalar=cfg.train.bbox_scalar):
        self.sclr = sclr
        self.bbx_loss_scalar= bbx_loss_scalar
    def __call__(self,bbx_pred, bbx_gdt, rpn_outside_weight, rpn_inside_weight):
        return mx.nd.sum(     rpn_outside_weight * mx.nd.smooth_l1(\
               data= rpn_inside_weight* (reshape_pred_bbox_ - gdt_bbox) ,\
                 scalar=cfg.train.l1_smooth_sclr) )*self.bbx_loss_scalar

class CfgLosser():
    """
        use Logistic Regression...
        Please use matrix multply before this operator!
    """
    def __init__(self):
        self.Loss = gluon.loss.LogisticLoss(label_format='binary')
    def __call__(self, cfg_pred, cfg_gdt, cfg_mask):
        """
            cfg_mask is provided by the iterator for balancing
        """
        return self.Loss(cfg_pred, cfg_gdt)*cfg_mask




def gen_symbol():
    """
                 generate full symbol, fixed parameter names are also returned
    """
    target_label = mx.sym.Variable('target_label')
    gdt_bbox         = mx.sym.Variable('target_bbox')
    rpn_outside_weight = mx.sym.Variable('rpn_outside_weight')
    rpn_inside_weight    = mx.sym.Variable('rpn_inside_weight')


    
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

#    score_mask = mx.sym.argmax(softmax_output, axis=-1, keepdims=True) # (b, HW, 1)

    symbol = mx.sym.Group([ softmax_output,  rpn_pred_bbox, rpn_bbox_loss])#, mx.sym.BlockGrad(symbol) ] )


    return symbol, fixed_param_names




