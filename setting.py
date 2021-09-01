from utils.header import *


def define_parser():
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    '''
    Setting Parameters 
    '''
    # Data가 들어있는 폴더 혹은 파일 지정
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

    # 주의: 윈도우 사이즈랑 똑같이 해야 돌아갑니다.
    args.batch_size = 72
    args.num_layers = 2

    args.epochs = 100

    return args
