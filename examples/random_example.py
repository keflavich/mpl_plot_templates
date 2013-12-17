import numpy as np
from mpl_plot_templates import imdiagnostics
import pylab as pl

arr = np.random.rand(1024/8,1024)

pl.figure(1)
pl.clf()
ax1 = imdiagnostics(arr)
pl.figure(2)
pl.clf()
ax2 = imdiagnostics(arr,square_aspect=True)

arr = np.random.rand(1024/256,1024)

pl.figure(3)
pl.clf()
ax3 = imdiagnostics(arr)
pl.figure(4)
pl.clf()
ax4 = imdiagnostics(arr,square_aspect=True)

pl.show()
