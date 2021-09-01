from .header import *


def train_plot(train, val):
    plt.plot(train, label='train loss')
    plt.plot(val, label='val loss')
    plt.legend()
    plt.show()


def predict_plot(y_true, y_val, view_count, method='forward'):
    if method == 'forward':
        plt.plot(y_true[:view_count], label='actual')
        plt.plot(y_val[:view_count], label='predict')
        plt.legend()
        plt.show()

    elif method == 'backward':
        plt.plot(y_true[view_count:], label='actual')
        plt.plot(y_val[view_count:], label='predict')
        plt.legend()
        plt.show()