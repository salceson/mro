# coding=utf-8

import os

__author__ = 'Michał Ciołczyk'

for k in [1, 5]:
    for cnn in ['1', '0']:
        for metric in ['euclid', 'mahalonobis']:
            print('k:', k, 'cnn:', cnn, 'metric:', metric)
            os.system('python knn.py ' + str(k) + ' ' + cnn + ' ' + metric + ' B')
