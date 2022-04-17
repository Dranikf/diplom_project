import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score

class ResultNet(nn.Module):
    '''Нейронная сеть в общем виде'''
    def __init__(self, neur_counts):
        super(ResultNet, self).__init__()
        # предполагаяется что выходной нейрон
        # всего один и пользователю не надо
        # его объявлять
        neur_counts = neur_counts.copy() + [1]
        
        # динамическое
        # добавление слоев
        layers = OrderedDict()
        for i in range(len(neur_counts) - 1):
            layers[str(i)] = (
                nn.Linear(
                    neur_counts[i],
                    neur_counts[i+1]
                )
            )
            # чтобы каждое обучение совпадало
            # надо, чтобы модели начинали обучение из
            # одной и той же точки - пусть это будут нули
            layers[str(i)].weight = \
            nn.Parameter(
                torch.rand(
                    layers[str(i)].weight.size()
                )
            )
            
            layers[str(i)].bias = \
            nn.Parameter(
                torch.rand(
                    layers[str(i)].bias.size()
                )
            )
            
            
        self.layers = nn.Sequential(layers)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            
        x = torch.sigmoid(self.layers[-1](x))
        return x
    

class My_data_set(Dataset):
    '''Набор данных'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return [self.X[idx,:], self.Y[idx, :]]

    
def get_loss_value(loss_fn, model, train_loader):
    '''Для всех данныз содержащихся 
    в переданном загрузчике просчитает
    переданную фнункцию потерь '''
    return loss_fn(
                model(train_loader.dataset.X),
                train_loader.dataset.Y
            ).item()

    
def train(
    model, optimizer, loss_fn, 
    train_loader, epochs=20, 
    lr_scheduler = None
):
    '''Алгоритм обучения сети'''
    # inputs:
    # model - модель которая подлежит обучению
    # optimizer - оптимизатор, который педполагается использовать
    # loss_fn - функция потерь
    # train_loader - загрузчик обучающих данных
    # epochs - эпохи используемые в нейронной сети
    # lr_scheduler - планировщик learning rate
    
    initial_loss = get_loss_value(
        loss_fn, model, train_loader
    )
    
    fun_arr = []
    fun_arr.append(initial_loss)
    
    for epoch in range(epochs):

        #model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
            
        # для отслеживания процесса
        # обучения буду сохранять текущее
        # значение целевой функии
        fun_arr.append(get_loss_value(
            loss_fn, model, train_loader
        ))

    
    return fun_arr


def plot_learning_curve(
    lc_data, start_show = 0, last_spes_show = 20
):
    '''Алгоритм обучения нейронной сети'''
    # inputs:
    # lc_data - собранная информация о обучении модели
    # start_show - та эпоха с которой следует начинать обучать
    # last_spes_show - та эпоха c конца с которой следует
    #                  начать отображение в правое окно
    
    plt.subplot(121)
    X_range = range(start_show, len(lc_data))

    plt.plot(X_range, lc_data[list(X_range)])
    plt.xlabel("Эпоха")
    plt.ylabel("Целевая функция")
    
    
    plt.subplot(122)
    X_range = range(
        len(lc_data)-last_spes_show, len(lc_data)
    )
    plt.plot(
        X_range, lc_data[list(X_range)]
    )
    plt.xlabel("Эпоха")


    
def models_fit_get_perfomance(
    hidden_layers_ranges, epochs, loss_fn,
    train_loader, X_test, y_test, weight_decay = 0.5, 
    lr = 0.1
):
    '''Метод обеспечивает построение моделей заданных форм
    и "снимает" инофрмацию о их свойсвах'''
    # inputs:
    # hidden_layers_ranges - инофрмация о внутренних
    #                        слоях модели как list
    #                        который содержит также списки
    #                        типа [<нейноны 1 слоя>, <нейноны 2 слоя>, ...]
    # lr -                   learning rate с которого начинается
    #                        оптимизация
    # epochs -               сколько эпох проходит каждая из 
    #                        форм модели
    # test_X, test_y -       данные на которых проводиться валидация
    #                        модели
    # train_loader -         загрузчик трерировочных данных
    # weight_decay -         параметр регуляризации
    # output:
    # [learning_info, auc_info, nets]
    # learning_info -        pandas.DataFrame по столбцам значения целевой
    #                        функции для эпохи в строке
    # auc_info -             pandas.Series AUC на тестовых данных для
    #                        каждой формы модели
    # nets -                 список с полученными моделями - с перспективой
    #                        дообучения
    
    
    # выходные массивы
    learning_info = pd.DataFrame()
    auc_info = pd.Series(dtype = 'float64')
    nets = []
    
    
    for layers_info in hidden_layers_ranges:
        # применение текущих настроек =================
        net = ResultNet(
            [train_loader.dataset.X.shape[1]] + 
            layers_info
        )
        optimizer = optim.Adam(
            net.parameters(), 
            weight_decay = weight_decay, 
            lr = lr
        )
        # применение текущих настроек =================
        # обучение ====================================
        #print(net.layers[0].bias)
        lc = train(
            net, optimizer, loss_fn, 
            train_loader, epochs = epochs
        )
        # обучение ====================================
        # снятие метрик ===============================
        probs_hat = net(
            torch.tensor(X_test.astype('float32'))
        ).detach().numpy()
    
        auc = roc_auc_score(y_test, probs_hat)
        # снятие метрик ===============================
        # сохранение информации =======================
        
        identyfyer = str(layers_info)
        learning_info.loc[:, identyfyer] = lc
        auc_info[identyfyer] = auc
        nets.append(net)
        
        # сохранение информации =======================
        print(identyfyer + " AUC = " + str(auc))
        
    return [learning_info, auc_info, nets]
