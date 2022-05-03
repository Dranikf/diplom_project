import numpy as np
from copy import deepcopy
from nets_algo import *

class model_trainer():
    '''Класс реализует алгоритм обучения сети'''
    def __init__(
        self, model, optimizer, 
        loss_fn, lr_scheduler = None
    ):
        # inputs:
        # model - модель которая подлежит обучению
        # optimizer - оптимизатор, который педполагается использовать
        # loss_fn - функция потерь
        # lr_shceduler - планировщих learning rate
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.train_loss = np.array([])
        self.test_loss = np.array([])
        
        # тут храним лучшую модель
        # из полученных если ошиентироваться
        # на ошибку на тестовых данных
        self.best_model = deepcopy(model)
        self.best_epoch = 0
        
        self.lr_scheduler = lr_scheduler
        
        
    def append_loses(self, train_loader, test_loader):
        '''Добавление ошибок рассчитанных
        на тренировочном и тестовом загрузчиках'''
        self.train_loss = np.append(
            self.train_loss,
            get_loss_value(
                self.loss_fn, 
                self.model, 
                train_loader
            )
        )
        self.test_loss = np.append(
            self.test_loss,
            get_loss_value(
                self.loss_fn, 
                self.model, 
                test_loader
            )
        )
        
    
    def fit(self, train_loader, test_loader,
            epochs = 20, check_epoch = 1):
        '''Провести тренировку модели'''
        # inputs:
        # train_loader - загрузчик тренировочных данных
        # test_loader - загрузчик тестовых данных
        # epochs - число эпох для обучения алгоритма
        # check_epoces - число эпох после чего алгорим
        #                может быть оставлен и ведется
        #                регистрация лучшей модели
        
        # получаем начальную ошибку до 
        # какого либо смещения весов
        self.append_loses(train_loader, test_loader)
        
        for epoch in range(epochs):

            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                output = self.model(inputs)
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # вычисление ошибок на тестовой и 
            # обучающих выборках на каждой эпохе
            self.append_loses(train_loader, test_loader)
            
        
            # работа с критериями остановки и
            # сохранения модели
            if epoch > check_epoch:
                # проверяем критерий остановки
                if self.test_loss[-1] > self.test_loss[-(check_epoch-1)]:
                    return
                
                # в том случае, если последняя полученная
                # ошибка на тестовых данных наименьшая
                # то нужно запомнить модель как наилучшую
                if self.test_loss[-1] == np.min(self.test_loss):
                    self.best_model = deepcopy(self.model)
                    self.best_epoch = epoch