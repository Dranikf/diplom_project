from sklearn.metrics import roc_curve


import numpy as np
import matplotlib.pyplot as plt

def get_p_bar(Y, probs_hat):
    fpr, tpr, thresholds = roc_curve(Y, probs_hat)

    # вот он p-штрих как максимальное
    return thresholds[np.argmax(np.abs(fpr - tpr))]


def draw_my_scatter(X1,X2,Y, legend_setting = ['класс1', 'класс2']):
    '''Отрисовка базового точечного графика с распределением по классам'''
    plt.scatter(X1[np.invert(Y)], X2[np.invert(Y)], color = 'blue')
    plt.scatter(X1[Y], X2[Y], color = 'red')

    plt.xlabel('$x_1$', fontsize = 15)
    plt.ylabel('$x_2$', fontsize = 15)
    plt.legend(legend_setting, fontsize = 14)