def grad_check(s):
    grad_sum = 0.
    for grads in mod._curr_module._exec_group.grad_arrays:
        for grad in grads:
            if grad is not None:
                grad_sum += np.sum( np.abs( grad.asnumpy() ) )
    logging.info('%s gradient sum: %f'%(s,grad_sum))
    return grad_sum


max_update_count=100
cnt=0
while cnt<max_update_count:
    mod.forward(d0)
    mod.backward()
    if grad_check('update[%d/%d]'%(cnt,max_update_count))<100:
        break
    mod.update()
    cnt += 1








