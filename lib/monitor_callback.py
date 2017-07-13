# -*- coding: utf-8 -*-

"""
Custom callback to expose train lass/accuracy information.
"""

import tflearn

class TrainMonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, train_service, data_size):
        self._loss = 0.0
        self._accuracy = 0.0
        self._epoch = 0
        self._iter = 0
        self._step = 0
        self._step_time = 0.0
        self._total_time = 0.0
        self._epoch_time = 0.0
        self._data_size = data_size
        self._train_service = train_service

    def on_batch_end(self, training_state, snapshot=False):
        self._loss = training_state.loss_value
        self._accuracy = training_state.acc_value
        self._epoch = training_state.epoch
        self._iter = training_state.current_iter
        self._step = training_state.step
        self._step_time = training_state.step_time_tota
        self._total_time = self._epoch_time + self._step_time
        self.__time = max(self._total_time, )
        #self.show()
        self.send_message()

    def on_epoch_end(self, training_state):
        self._epoch_time = self._epoch_time + training_state.step_time_total

    def send_message(self):
        self._train_service.sendTrainMeticsMessage(self._iter, self._data_size, self._loss)
        self._train_service.sendTestMetricsMessage(self._iter, self._loss, self._accuracy)

    def show(self):
        print("data_size: %d" % self._data_size)
        print("loss: %.2f" % self._loss)
        print("accuracy: %.2f" % self._accuracy)
        print("epoch: %d" % self._epoch)
        print("iter: %d" % self._iter)
        print("step: %d" % self._step)
        print("step time: %.2f" % self._step_time)
