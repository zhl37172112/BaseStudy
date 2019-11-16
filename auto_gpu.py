import torch


def auto_gpu(*args):
    ret_args = []
    if torch.cuda.is_available():
        for arg in args:
            arg = arg.cuda()
            ret_args.append(arg)
    else:
        ret_args = args
    if len(ret_args) == 1:
        ret_args = ret_args[0]
    return ret_args