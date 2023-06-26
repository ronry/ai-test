# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../../data/lena.png') #读入图像
plt.imshow(img)

plt.show()