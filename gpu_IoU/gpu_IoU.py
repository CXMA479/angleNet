"""
too many lines, setup a new module

Chen Y.liang
Nov 22, 2017
"""
import mxnet as mx
import numpy as np


def mx_rotate(x,y, alpha, O=(0.,0.)):
    #TODOK:
#       1.{O} is to accept array-shape input for allowing row-wise rotation \/
#       2.{x,y} are to allow (1,n) shape input for supporting broadcast-rotate w.r.t alphas (naturally realized by mx.nd.broadcast_mul)
    """
        args:
            x,y: mx.nd.array
                ({Batch/1}, n)  vector shape is supported for single gdt rotation
            alpha: mx.nd.array
                (Batch,1), measured by RAD
            O: coordinate of rotation origin:
                TWO elements tuple:
                    two elements of { scalar(default), mx.nd.array with shape of (Batch,1) }

        return:
            rot_x, rot_y   | (Batch, n), (Batch, n)
    """
    """y
        |     P(x,y)
        |   /
        |  /
        | /
        |/alpha   P0'
      O'|--------------->x
        |\angle1
        | \
        |  \
            \
            P0(x0,y0)

        1. with O' as origin:
            1. return {P0} to x axes as {P0'} through {angle1}
            2. rotate P0' to P' through {alpha}
        2. shift the system with {O'}

    """
    def raw_rotate(x,y,alpha):
        """
            see rotation matrix @ https://en.wikipedia.org/wiki/Rotation_matrix

            R(\theta) =[cos  -sin
                        sin   cos]

            [x' y']^T  =R(\theta)[x  y]^T

            args:
                x,y: mx.nd.array
                    (Batch, k^2)
                alpha: mx.nd.array
                    (Batch, 1)        RAD
        """

        [sin_alpha, cos_alpha] = [mx.nd.sin(alpha), mx.nd.cos(alpha) ]
        x_new = mx.nd.broadcast_mul(x, cos_alpha ) - mx.nd.broadcast_mul(y, sin_alpha)
        y_new = mx.nd.broadcast_mul(x, sin_alpha ) + mx.nd.broadcast_mul(y, cos_alpha)
        return x_new, y_new

    # check O's shape...
    assert len(O)==2 and isinstance(O,tuple), len(O)
    # trans to O' new system
    [x_new, y_new] = [ _ - o for _,o in zip([x, y],O) ]

    ## to x axes
    ### get angle1...
    angle1 = mx.nd.arctan(y_new/(x_new + 1e-12) )
    ### ! remember negtive
    x_new, y_new = raw_rotate(x_new, y_new, - angle1)

    ## to P'
    x_new, y_new = raw_rotate(x_new, y_new, alpha+angle1)

    # transfer back to O system
    [x_new, y_new] = [_ + o for _, o in zip([x_new, y_new],O) ]

    return x_new, y_new

def __anchor2limits__(anchor_alpha, mx_anchor_rh, mx_anchor_rw, mx_anchor_x, mx_anchor_y):
    """
        aims to provide limits in one's OWN axis for each anchor
        args:
            anchors: np.array (num, 5)

        arg(s):
            anchor: np.array (num, 5)
                x, y, alpha, rh, rw
        return:
            limits_x_l, limit_x_r,\
            limits_y_t, limit_y_b   : mx.nd.array  (num, 1)

        y
        |  /rh
        | /
        |/alpha
        |---------------x
        |\  alpha - pi/2
        | \
        |  \ rw
    """
    alpha = anchor_alpha
    rh, rw = [mx_anchor_rh, mx_anchor_rw]
    x, y = [mx_anchor_x, mx_anchor_y]
    ## use vector operation
    rhx, rhy = [rh*   F(alpha)          for F in [mx.nd.cos, mx.nd.sin] ]
    rwx, rwy = [rw*F( alpha - np.pi/2 ) for F in [mx.nd.cos, mx.nd.sin] ]

    # okay, four vertices are on the way...
    ## I quadrant comes firstly, clock-wise oder...
    ### \bm{v1} = \bm{P} + \bm{rh}    - \bm{rw}
    bp_x1, bp_y1 = [  P  +     rh_d   -     rw_d      for P, rh_d, rw_d\
                in zip([x, y], [rhx, rhy], [rwx, rwy]) ]

    ## IV sector
    ### \bm{v2} = \bm{P} + \bm{rh}    + \bm{rw}
    bp_x2, bp_y2 = [  P  +     rh_d   +     rw_d      for P, rh_d, rw_d\
                in zip([x, y], [rhx, rhy], [rwx, rwy]) ]

    ## III box
    ### \bm{v3} = \bm{P} - \bm{rh}    + \bm{rw}
    bp_x3, bp_y3 = [  P  -     rh_d   +     rw_d      for P, rh_d, rw_d\
                in zip([x, y], [rhx, rhy], [rwx, rwy]) ]

    ## II's turn
    ### \bm{v4} = \bm{P} - \bm{rh}    - \bm{rw}
    bp_x4, bp_y4 = [  P  -     rh_d   -     rw_d      for P, rh_d, rw_d\
                in zip([x, y], [rhx, rhy], [rwx, rwy]) ]

    # concatenate together...
    bp_xs, bp_ys = [ mx.nd.concat(a1,a2,a3,a4,dim=1) for a1,a2,a3,a4 in\
            zip([bp_x1, bp_y1],\
                [bp_x2, bp_y2],\
                [bp_x3, bp_y3],\
                [bp_x4, bp_y4])  ]
#    assert 0, (bp_xs, bp_ys)
    ## rotate to each anchor's own coordinate system...
    x_limits, y_limits = mx_rotate(bp_xs, bp_ys,  - alpha)#, (x, y) ) # negative angle!
#    assert 0, (x_limits, y_limits)
    ## detect the limits...
        # in this section, axes aligns to Image axes(i.e. top means minima of y)
        # dims should be kept for further broadcast operation(s)
    x_limits_l, y_limits_t = [ mx.nd.min(limits, axis=1, keepdims=True)\
                    for limits in [x_limits, y_limits] ]
    x_limits_r, y_limits_b = [ mx.nd.max(limits, axis=1, keepdims=True)\
                    for limits in [x_limits, y_limits] ]

    return x_limits_l, x_limits_r, y_limits_t, y_limits_b

def __scater_fill__(gdt_alpha, mx_gdt_rh, mx_gdt_rw, mx_gdt_x, mx_gdt_y, k, ctx):
    """
        fill the gdt boxes with uniform-distributed points
        args:
            gdt: (n,5) np.Array
    """
    assert gdt_alpha.shape[1] ==1, gdts.shape
    n = gdt_alpha.shape[0]
    alpha = gdt_alpha
    rh, rw = [mx_gdt_rh, mx_gdt_rw]
    x_offset, y_offset = [mx_gdt_x, mx_gdt_y]


    # s1: grid...
    ## do computations in a small scale as much as poosible...
    x=np.arange(0,k)*1./k -0.5 + 1./(2*k) # align center to origin for rotation(-.5) and, each node represents a integritied block(+ 1/(2k))
    X,Y = np.meshgrid(x,x)
    x_grid, y_grid = [ np.reshape(_,(1,k**2)) for _ in [X,Y] ]
    # repeat to same shape[0] of gdt...
    x_grid, y_grid = [ mx.nd.array(np.repeat(_, n, axis=0), ctx) for _ in [x_grid, y_grid]  ]

    ## s2: scale...
    x_grid, y_grid = [mx.nd.broadcast_mul(grid, 2*r) for grid,r in zip([x_grid, y_grid], [rh, rw])  ]

    ## s3: rotate...
        #  do it in a seperate definition
    x_grid, y_grid = mx_rotate(x_grid, y_grid, alpha)

    ## s4: shift...
#    assert 0, (x_grid.shape, x_offset.shape)
    x_grid, y_grid = [ mx.nd.broadcast_add(_ ,  offset) for\
            _, offset in zip([x_grid, y_grid], [x_offset, y_offset] )  ]

    return x_grid, y_grid



def gpu_it_IoU(anchors, gdts, k, ctx=mx.gpu()):
    """
        A GPU version of it_IoU (c. cytool.pyx for {it_IoU} ) is needed.
            cause not only in feedforward training but also in prediction phase, it is required,
                also, there are uncertain number of tricks to debug...
        Thanks to MXNet's engine -_-

        i will try...

        Chen Y.liang
        Nov 16, 2017

        args:
            anchors: np.array (num,5) : x, y, alphaR, rh, rw

            gdts:    ~        (n, 5)

            k: int  partition number along one axes

        return: np.array
            iou_matrix: (num,n)

        by contrasting the conventioal one, there is no need for {has_intesec} since we are now
            backed by the mammoth computing...
    """
#    assert 0,anchors.shape
    anchor_rh, anchor_rw = [ mx.nd.reshape( mx.nd.array(anchors[:,i], ctx), shape=(-1,1)) for i in [3,4]  ]
    gdt_rh, gdt_rw = [mx.nd.reshape( mx.nd.array(gdts[:,i], ctx ),shape=(-1,1)) for i in [3,4]  ]

    gdt_alpha = mx.nd.reshape(mx.nd.array(gdts[:,2], ctx), shape=(-1,1))
    anchor_alpha = mx.nd.reshape( mx.nd.array(anchors[:,2], ctx), shape=(-1,1) )

    gdt_x, gdt_y = [mx.nd.reshape(mx.nd.array(gdts[:,i], ctx), shape=(-1,1) ) for i in [0, 1] ]
    anchor_x, anchor_y =  [ mx.nd.reshape( mx.nd.array(anchors[:,i], ctx), shape=(-1,1)  ) for i in [0,1] ]

    num = anchors.shape[0]
    n = gdts.shape[0]
    iou_matrix_T = mx.nd.empty((n,num))


    # s1: scater fill gdt...
    gdt_sc_xs, gdt_sc_ys = __scater_fill__(gdt_alpha, gdt_rh, gdt_rw, gdt_x, gdt_y, k, ctx) # return (n, k^2), (n, k^2)
    # s2: convert2vertice...
#    anchor_bp_alpha = mx.nd.reshape(mx.nd.array(anchors[:,2], ctx),shape=(-1,1) ) # (num, 1)

    ## for anchor limit, first returned is minima...
    anchor_limit_x_l,anchor_limit_x_r,\
    anchor_limit_y_t,anchor_limit_y_b = __anchor2limits__(anchor_alpha, anchor_rh, anchor_rw,\
                                            anchor_x, anchor_y)   # each return is (num,1)
#    assert 0, (anchor_limit_x_l, anchor_limit_x_r, anchor_limit_y_t, anchor_limit_y_b)
    # s3: calculate areas for both anchor and gdt
    A_anchor, A_gdt = [4*h*w for h,w in zip([anchor_rh, gdt_rh], [anchor_rw, gdt_rw]) ]
    # s4: okay, let's loop...
    for gdt_idx in xrange(n):
        gdt_sc_x, gdt_sc_y = [ mx.nd.reshape(a, shape=(1,-1)) for a \
                in [ gdt_sc_xs[gdt_idx], gdt_sc_ys[gdt_idx] ] ]   # (1, k^2)
        rot_gdt_sc_x, rot_gdt_sc_y = mx_rotate(gdt_sc_x, gdt_sc_y, - anchor_alpha) # (num, k^2)
#        assert 0, (rot_gdt_sc_x, rot_gdt_sc_y)
        # begin comparing...
        in_x = mx.nd.broadcast_greater(rot_gdt_sc_x, anchor_limit_x_l)\
                * mx.nd.broadcast_lesser(rot_gdt_sc_x, anchor_limit_x_r)

        in_y = mx.nd.broadcast_greater(rot_gdt_sc_y, anchor_limit_y_t)\
                * mx.nd.broadcast_lesser(rot_gdt_sc_y, anchor_limit_y_b)

        in_p      = in_y * in_x    # (num, k^2)
#        assert 0, in_p
#        assert 0, (rot_gdt_sc_x, rot_gdt_sc_y, in_p, in_p.shape)
        in_p      = mx.nd.sum(in_p, axis=1, keepdims=True)/k**2 # (num, 1)
#        assert 0, in_p#.shape
        area_in = in_p * A_gdt[gdt_idx]
#        assert 0, area_in.shape
#        assert 0, iou_matrix_T.shape
        iou_matrix_T[gdt_idx] =mx.nd.reshape(area_in /( mx.nd.broadcast_add(A_gdt[gdt_idx], A_anchor) - area_in), shape=(-1,))
#

    return iou_matrix_T.T.asnumpy()#, gdt_sc_xs, gdt_sc_ys






