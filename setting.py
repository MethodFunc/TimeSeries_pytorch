from utils.header import *


def define_parser():
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    '''
    Setting Parameters 
    '''
    args.path = './data'

    # data setting
    args.window_size = 72
    args.sliding_func = True

    # support: minmax, normal, robust, standard
    args.method = 'minmax'

    # Torch setting
    # support mse, huber
    args.loss_fn = 'mse'

    # support adam, rmsprop, sgd
    args.optim = 'adam'

    args.lr = 1e-3

    args.hidden_size = 128
    args.output = 1

    args.batch_size = 72
    args.num_layers = 2

    args.epochs = 100

    return args
