#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:41:04 2021

@author: hossein
"""

import skimage.io
import matplotlib.pyplot as plt


img_path = '/home/hossein/deep-person-reid/datasets/market1501/Market-1501-v15.09.15/query/0017_c4s1_002051_00.jpg'
img = skimage.io.imread(img_path)

# Cut your window in 1 row and 2 columns, and start a plot in the first part
fig , ax = plt.subplots(nrows=1,ncols=2)
  #Show images
ax.ravel()[0].imshow(img)
ax.ravel()[0].set_axis_off()

ax.ravel()[1].text(0, 0.8, 'Woman', style='italic', fontsize='xx-large')
ax.ravel()[1].text(1, 0.8, 'Handbag', style='italic', fontsize='xx-large')

ax.ravel()[1].text(0.4, 0.7, 'Knot Hair', style='italic', fontsize='xx-large')

ax.ravel()[1].text(0.3, 0.55, 'White Shirt', style='italic', fontsize='xx-large')
ax.ravel()[1].text(0.45, 0.59, 'Simple', style='italic', fontsize='xx-large')
# ax.ravel()[1].text(0.4, 0.5, 'leg', style='italic', fontsize='xx-large')

ax.ravel()[1].text(0.3, 0.4, 'Green Shorts', style='italic', fontsize='xx-large')

ax.ravel()[1].text(0.3, 0.3, 'White shoes', style='italic', fontsize='xx-large')

ax.ravel()[1].set_axis_off()
plt.show()




