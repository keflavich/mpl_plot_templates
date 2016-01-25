import pylab as pl
#import mpl_toolkits.axes_grid.parasite_axes as mpltk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import asinh_norm
import numpy as np

def imdiagnostics(data, axis=None, square_aspect=False, percentiles=None,
                  second_xaxis=None):
    """
    Create a 'waterfall plot' of (timestream) data

    Parameters
    ----------
    second_xaxis : None or `numpy.ndarray`
        An array of shape data.shape[1] describing the X-axis
    """
    if axis is None:
        fig = pl.gcf()
        axis = pl.gca()
        #axis = mpltk.HostAxes(fig, (0.1,0.1,0.7,0.7), adjustable='datalim')
    else:
        fig = axis.get_figure()

    bad = np.isnan(data)
    if bad.any():
        dstd = data[~bad].std()
        dmean = data[~bad].mean()
    else:
        dstd = data.std()
        dmean = data.mean()

    if square_aspect:
        im = axis.imshow(data,vmin=dmean-5*dstd,vmax=dmean+5*dstd,norm=asinh_norm.AsinhNorm(),
                         aspect=data.shape[1]/float(data.shape[0]))
    else:
        im = axis.imshow(data,vmin=dmean-5*dstd,vmax=dmean+5*dstd,norm=asinh_norm.AsinhNorm(),
                         aspect='auto')

    
    # debug; not needed
    # xlim,ylim =axis.get_xlim(),axis.get_ylim()

    divider = make_axes_locatable(axis)
    divider.set_aspect(False)  # MAGIC!!!!! False good, True bad, nothing BAD.  Wtf?
    #divider.get_horizontal()[0]._aspect=0.5
    
    right = divider.append_axes("right", size="15%", pad=0.05)
    #right
    #right = divider.new_horizontal(size="15%", pad=0.05, sharey=axis)
    #fig.add_axes(right)
    vright = divider.append_axes("right", size="15%", pad=0.05)
    #vright = divider.new_horizontal(size="15%", pad=0.05, sharey=axis)
    #fig.add_axes(vright)
    if percentiles is not None:
        meany = np.array([np.percentile(data, p, axis=1) for p in percentiles])
    else:
        meany = data.mean(axis=1)
    erry = data.std(axis=1)

    right.plot(meany.T,np.arange(data.shape[0]))
    right.set_ylim(0,data.shape[0]-1)
    right.set_yticks([])
    if meany.max() > meany.min():
        right.set_xticks([meany.min(),(meany.max()+meany.min())/2.,meany.max()])
    else:
        right.set_xticks([])
    pl.setp(right.xaxis.get_majorticklabels(), rotation=70)
    right.set_title("$\mu$")

    vright.plot(erry,np.arange(erry.size))
    vright.set_ylim(0,erry.size-1)
    vright.set_yticks([])
    if erry.max()>erry.min():
        vright.set_xticks([erry.min(),(erry.max()+erry.min())/2.,erry.max()])
    else:
        vright.set_xticks([])
    vright.set_xlabel("$\sigma$")
    vright.xaxis.set_ticks_position('top')
    pl.setp(vright.xaxis.get_majorticklabels(), rotation=70)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)

    top = divider.append_axes("top", size="15%", pad=0.05)
    vtop = divider.append_axes("top", size="15%", pad=0.05)
    #top = divider.new_vertical(size="15%", pad=0.05, sharex=axis)
    #vtop = divider.new_vertical(size="15%", pad=0.05, sharex=axis)
    #fig.add_axes(top)
    #fig.add_axes(vtop)

    if percentiles is not None:
        meanx = np.array([np.percentile(data, p, axis=0) for p in percentiles])
    else:
        meanx = data.mean(axis=0)

    errx = data.std(axis=0)
    top.plot(np.arange(data.shape[1]),meanx.T)
    top.set_xlim(0,data.shape[1]-1)
    top.set_xticks([])
    if meanx.max()>meanx.min():
        top.set_yticks([meanx.min(),(meanx.max()+meanx.min())/2.,meanx.max()])
    else:
        top.set_yticks([])
    pl.setp(top.yaxis.get_majorticklabels(), rotation=20)
    top.set_title("$\mu$")
    vtop.plot(np.arange(errx.size),errx,)
    vtop.set_xlim(0,errx.size-1)
    if errx.max()>errx.min():
        vtop.set_yticks([errx.min(),(errx.max()+errx.min())/2.,errx.max()])
    else:
        vtop.set_yticks([])
    vtop.set_ylabel("$\sigma$")
    pl.setp(vtop.yaxis.get_majorticklabels(), rotation=-20)

    if second_xaxis is not None:
        vtop.xaxis.set_ticks_position('top')
        vtop.xaxis.set_label_position('top')
        inds = np.linspace(0, errx.size, 8).astype('int')
        #inds = axis.xaxis.get_ticklocs()
        inds = inds[(inds>0) & (inds<errx.size)].astype('int')
        vtop.xaxis.set_ticks(inds)
        vtop.xaxis.set_ticklabels(["{0:0.4g}".format(x) for x in second_xaxis[inds]])
        vtop.set_xlim(0,errx.size-1)
    else:
        vtop.set_xticks([])

    # debug; not needed
    #axis.set_xlim(xlim)
    #axis.set_ylim(ylim)

    return axis
