import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import asinh_norm
import numpy as np

def imdiagnostics(data, axis=pl.gca()):
    dstd = data.std()
    dmean = data.mean()
    im = axis.imshow(data,vmin=dmean-5*dstd,vmax=dmean+5*dstd,norm=asinh_norm.AsinhNorm(),aspect=data.shape[1]/float(data.shape[0]))
    divider = make_axes_locatable(axis)

    right = divider.append_axes("right", size="15%", pad=0.05)
    vright = divider.append_axes("right", size="15%", pad=0.05)
    meany = data.mean(axis=1)
    erry = data.std(axis=1)

    right.plot(meany,np.arange(meany.size))
    right.set_ylim(0,meany.size-1)
    right.set_yticks([])
    right.set_xticks([meany.min(),(meany.max()+meany.min())/2.,meany.max()])
    pl.setp(right.xaxis.get_majorticklabels(), rotation=70)
    right.set_title("$\mu$")

    vright.plot(erry,np.arange(erry.size))
    vright.set_ylim(0,erry.size-1)
    vright.set_yticks([])
    vright.set_xticks([erry.min(),(erry.max()+erry.min())/2.,erry.max()])
    vright.set_xlabel("$\sigma$")
    vright.xaxis.set_ticks_position('top')
    pl.setp(vright.xaxis.get_majorticklabels(), rotation=70)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    top = divider.append_axes("top", size="15%", pad=0.05)
    vtop = divider.append_axes("top", size="15%", pad=0.05)
    meanx = data.mean(axis=0)
    errx = data.std(axis=0)
    top.plot(np.arange(meanx.size),meanx)
    top.set_xlim(0,meanx.size-1)
    top.set_xticks([])
    top.set_yticks([meanx.min(),(meanx.max()+meanx.min())/2.,meanx.max()])
    pl.setp(top.yaxis.get_majorticklabels(), rotation=20)
    top.set_title("$\mu$")
    vtop.plot(np.arange(errx.size),errx,)
    vtop.set_xlim(0,errx.size-1)
    vtop.set_xticks([])
    vtop.set_yticks([errx.min(),(errx.max()+errx.min())/2.,errx.max()])
    vtop.set_ylabel("$\sigma$")
    pl.setp(vtop.yaxis.get_majorticklabels(), rotation=-20)
