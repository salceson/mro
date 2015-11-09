# coding=utf-8

import os

__author__ = 'Michał Ciołczyk'

dir = 'selfie/input/'

for i in range(30):
    os.system('convert -resize 30x50! -colorspace Gray ' + dir + str(i) +
              '-in.png ' + dir + str(i) + '.jpg')
