import matplotlib as mpl
import pylab as pl
import numpy as np

def multidigitize(x,y,binsx,binsy):
    dx = np.digitize(x.flat, binsx)
    dy = np.digitize(y.flat, binsy)
    return dx,dy

def linlogspace(xmin,xmax,n):
    return np.logspace(np.log10(xmin),np.log10(xmax),n)

def adaptive_param_plot(x,y,
                        bins=10,
                        threshold=5,
                        marker='.',
                        marker_color=None,
                        ncontours=5,
                        levels=None,
                        fill=False,
                        mesh=False,
                        contourspacing=linlogspace,
                        mesh_alpha=0.5,
                        norm=None,
                        axis=None,
                        cmap=None,
                        percentilelevels=None,
                        **kwargs):
    """
    Plot contours where the density of data points to be plotted is too high

    Many of these parameters are just passed on to matplotlib's "plot" and
    "contour" functions

    Parameters
    ----------
    bins: int or ndarray
        The number of bins or a list of bins.  Passed to np.histogram2d; see
        their docs for details
    threshold: int
        The minimum number of points to replace a bin with a contour.  For
        npoints<=threshold, the individual points will be plotted
    marker: str
        Any valid marker (see matplotlib.plot)
    marker_color: str
        Any valid marker color
    ncontours: int
        Number of contour levels
    levels: None or list 
        Optional override for automatically computed levels
    fill: bool
        Use filled contours?
    mesh: bool
        Use a color mesh instead of contours (i.e., filled pixels)
    contourspacing: function
        A function to determine the contour spacing.  Should be 
        np.linspace or the linlogspace function
        defined above.  The function must accept arguments of:
        lowest contour level, highest contour level, number of contours
    mesh_alpha: float
        Alpha opacity parameter of the colormesh, if used
    norm: matplotlib.colors.Normalize
        A normalization to use for the contours or colormesh
    axis: None or matplotlib.Axes
        A matplotlib axis to plot on
    cmap: matplotlib colormap
        A valid matplotlib color map
    kwargs: dict
        Passed to plot, contour, AND colormesh, so must be valid for ALL 3!
    """
    
    if axis is None:
        axis = pl.gca()

    ok = np.isfinite(x) * np.isfinite(y)

    # set the number of bins to be an integer value, which can be extracted
    # from an array-style bin set
    if hasattr(bins,'ndim') and bins.ndim == 2:
        # If you define the bins as an array, they define the BIN EDGES, so nbins=len(bins)-1
        nbinsx,nbinsy = bins.shape[1]-1,bins.shape[1]-1
    else:
        try:
            nbinsx = nbinsy = len(bins)-1
        except TypeError:
            nbinsx = nbinsy = bins

    H,bx,by = np.histogram2d(x[ok],y[ok],bins=bins)

    # determine the locations of each pixel
    dx,dy = multidigitize(x[ok],y[ok],bx,by)

    # anything beyond the range of the histogram bins defaults to plottable=True
    # need +1 because anything <bins.min() or >bins.max() is on the edges...
    plottable = np.ones([nbinsx+2,nbinsy+2], dtype='bool')
    # Need a cropped version of "plottable" to index H
    # This is a view on plottable, so should result in inplace modification...
    # (this version does not include the points > bins.max() and < bins.min(),
    # which will all be plotted)
    plottable_hist = plottable[1:-1,1:-1]
    assert H.shape == plottable_hist.shape
    # points are plottable if below the threshold
    plottable_hist[H > threshold] = False

    #H[plottable] = np.nan
    H[plottable_hist] = 0
    #H = np.ma.masked_where(plottable,H)
    toplot = plottable[dx,dy]

    cx = (bx[1:]+bx[:-1])/2.
    cy = (by[1:]+by[:-1])/2.
    if levels is None:
        if percentilelevels is not None:
            sortedH = np.sort(H.flat)
            cumfrac = np.cumsum(sortedH) / H.sum()
            levels = [sortedH[np.argmin(np.abs(cumfrac - plev))] for plev in percentilelevels]
        else:
            levels = contourspacing(threshold-0.5,H.max(),ncontours)
    #levels = contourspacing(0,H.max(),ncontours)

    if cmap is None:
        cmap = mpl.cm.get_cmap()
        cmap.set_under((0,0,0,0))
        cmap.set_bad((0,0,0,0))

    if fill:
        con = axis.contourf(cx,cy,H.T,levels=levels,norm=norm,cmap=cmap,**kwargs)
    else:
        con = axis.contour(cx,cy,H.T,levels=levels,norm=norm,cmap=cmap,**kwargs)
    if mesh:
        mesh = axis.pcolormesh(bx,by,H.T, **kwargs)
        mesh.set_alpha(mesh_alpha)
    
    if 'linestyle' in kwargs:
        kwargs.pop('linestyle')

    if marker not in ('none', None):
        axis.plot(x[ok][toplot],
                  y[ok][toplot],
                  linestyle='none',
                  marker=marker,
                  markerfacecolor=marker_color,
                  markeredgecolor=marker_color,
                  **kwargs)

    return cx,cy,H,x[ok][toplot],y[ok][toplot]
