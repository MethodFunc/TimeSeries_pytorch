__all__ = ['scaling', 'DataEval']

from .header import *

"""
Scaling and metrics
auth: Methodfunc - Kwak Piljong
date: 2021.08.26
modify date: 2021.08.27
version: 0.2
"""


def scaling(data, method='minmax'):
    """
    select method : minmax, normal, robust, standard
    """

    if method == 'minmax':
        sc = MinMaxScaler()

    elif method == 'robust':
        sc = RobustScaler()

    elif method == 'normal':
        sc = Normalizer()

    elif method == 'standard':
        sc = StandardScaler()

    else:
        raise 'Not support scale'

    sc_data = sc.fit_transform(data)

    return sc, sc_data


class DataEval:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        self.y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        self.MSE = 0
        self.RMSE = 0
        self.MAE = 0
        self.MAPE = 0

    def calc(self):
        self.__mse()
        self.__rmse()
        self.__mae()
        self.__mape()
        print('Low then Better')
        print(f"Mean squared error                : {self.MSE:.4f}")
        print(f"Root mean squared error           : {self.RMSE:.4f}")
        print(f"Mean absolute error               : {self.MAE:.4f}")
        print(f"Mean absolute percentage error    : {self.MAPE:.4f}")
        print('====================================================')
        self.__check_acc()

    def __mse(self):
        self.MSE = np.mean(np.square(np.subtract(self.y_true, self.y_pred)))

    def __rmse(self):
        self.RMSE = np.sqrt(np.mean(np.square(np.subtract(self.y_true, self.y_pred))))

    def __mae(self):
        self.MAE = np.mean(np.abs(np.subtract(self.y_true, self.y_pred)))

    def __mape(self):
        self.MAPE = np.mean(np.abs(np.subtract(self.y_true, self.y_pred) / self.y_true)) * 100

    def __check_acc(self):
        """
        calc acc
        """
        y_true = self.y_true.ravel()
        y_pred = self.y_pred.ravel()
        error_rate = ((np.abs(y_pred - y_true) / y_true) * 100)
        calc_acc = np.array([100 - i for i in error_rate if i != np.inf])

        acc = np.zeros(calc_acc.shape)

        for i in range(len(calc_acc)):
            if calc_acc[i] < 0:
                acc[i] = 0
            else:
                acc[i] = calc_acc[i]

        print(f"Acc Mean     : {np.mean(acc):.4f}%")
        print(f"Acc Max      : {np.max(acc):.4f}%")
        print(f"Acc Min      : {np.min(acc):.4f}%")
