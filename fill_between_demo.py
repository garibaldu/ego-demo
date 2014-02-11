#!/usr/bin/env python
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
import numpy as np

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)

fig = figure()
ax1 = fig.add_subplot(111)

myFill = ax1.fill_between(x, y1, y2)


fig.__dict__


#myFill.set_verts()

show()

