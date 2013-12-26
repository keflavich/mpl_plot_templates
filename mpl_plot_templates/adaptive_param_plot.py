import matplotlib as mpl
import pylab as pl
import numpy as np

def multidigitize(x,y,binsx,binsy):
    dx = np.digitize(x.flat, binsx)
    dy = np.digitize(y.flat, binsy)
    return dx,dy

def linlogspace(xmin,xmax,n):
    return np.logspace(np.log10(xmin),np.log10(xmax),n)

def adaptive_param_plot(x,y,bins=10,threshold=5,
                        marker='.',
                        marker_color=None,
                        ncontours=5,
                        fill=False,
                        mesh=False,
                        contourspacing=linlogspace,
                        mesh_alpha=0.5,
                        norm=None,
                        **kwargs):
    """
    Plot contours where the density of data points to be plotted is too high
    """

    ok = np.isfinite(x) * np.isfinite(y)

    H,bx,by = np.histogram2d(x[ok],y[ok],bins=bins)

    dx,dy = multidigitize(x[ok],y[ok],bx,by)
    dx[dx==bins+1] = bins
    dy[dy==bins+1] = bins

    plottable = H <= threshold
    #H[plottable] = np.nan
    H[plottable] = 0
    #H = np.ma.masked_where(plottable,H)
    toplot = plottable[dx-1,dy-1]

    cx = (bx[1:]+bx[:-1])/2.
    cy = (by[1:]+by[:-1])/2.
    levels = contourspacing(threshold-0.5,H.max(),ncontours)
    #levels = contourspacing(0,H.max(),ncontours)

    cm = mpl.cm.get_cmap()
    cm.set_under((0,0,0,0))
    cm.set_bad((0,0,0,0))

    if fill:
        con = pl.contourf(cx,cy,H.T,levels=levels,norm=norm,**kwargs)
    else:
        con = pl.contour(cx,cy,H.T,levels=levels,norm=norm,**kwargs)
    if mesh:
        mesh = pl.pcolormesh(bx,by,H.T, **kwargs)
        mesh.set_alpha(mesh_alpha)
    
    if 'linestyle' in kwargs:
        kwargs.pop('linestyle')

    pl.plot(x[ok][toplot],
            y[ok][toplot],
            linestyle='none',
            marker=marker,
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            **kwargs)

    return cx,cy,H,x[ok][toplot],y[ok][toplot]
