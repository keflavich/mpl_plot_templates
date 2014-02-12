from mpl_plot_templates import adaptive_param_plot
import numpy as np
import pylab as pl

pl.clf()

adaptive_param_plot(np.random.randn(1000)+np.linspace(0,5,1000),np.random.randn(1000)+np.linspace(0,5,1000), fill=True, alpha=0.5, threshold=10)

pl.savefig('param_plot_example.png')
