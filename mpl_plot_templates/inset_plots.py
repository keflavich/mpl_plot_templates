from astropy import units as u
from astropy import coordinates
from astropy.stats import mad_std
from astropy.visualization import simple_norm
import astropy.visualization
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import radio_beam
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector, Path
from matplotlib.transforms import Bbox
from matplotlib.patches import Polygon, PathPatch, ConnectionPatch

def mark_inset_otherdata(axins, parent_ax, bl, tr, loc1, loc2, edgecolor='b'):

    frame = parent_ax.wcs.wcs.radesys.lower()
    frame = ax.wcs.world_axis_physical_types[0].split(".")[1]

    blt = bl.transform_to(frame)
    trt = tr.transform_to(frame)
    (rx1,ry1),(rx2,ry2) = (parent_ax.wcs.wcs_world2pix([[blt.spherical.lon.deg,
                                                         blt.spherical.lat.deg]],0)[0],
                           parent_ax.wcs.wcs_world2pix([[trt.spherical.lon.deg,
                                                         trt.spherical.lat.deg]],0)[0]
                          )
    bbox = Bbox(np.array([(rx1,ry1),(rx2,ry2)]))
    rect = TransformedBbox(bbox, parent_ax.transData)

    markinkwargs = dict(fc='none', ec=edgecolor)

    pp = BboxPatch(rect, fill=False, **markinkwargs)
    parent_ax.add_patch(pp)

    p1 = BboxConnector(axins.bbox, rect, loc1=loc1, **markinkwargs)
    axins.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(axins.bbox, rect, loc1=loc2, **markinkwargs)
    axins.add_patch(p2)
    p2.set_clip_on(False)

    return bbox, rect

def mark_inset_otherdata_full(axins, parent_ax, data, loc1=1, loc2=3, edgecolor='b'):
    bl = axins.wcs.pixel_to_world(0, 0)
    tr = axins.wcs.pixel_to_world(*data.shape[::-1]) # x,y not y,x

    return mark_inset_otherdata(axins, parent_ax, bl, tr, loc1=loc1, loc2=loc2, edgecolor=edgecolor)

def mark_inset_generic(axins, parent_ax, data, loc1=1, loc2=3, edgecolor='b'):
    bl = axins.wcs.pixel_to_world(0, 0)
    br = axins.wcs.pixel_to_world(0, data.shape[0])
    tl = axins.wcs.pixel_to_world(data.shape[1], 0) # x,y not y,x
    tr = axins.wcs.pixel_to_world(*data.shape[::-1]) # x,y not y,x

    frame = parent_ax.wcs.wcs.radesys.lower()
    frame = ax.wcs.world_axis_physical_types[0].split(".")[1]

    blt = bl.transform_to(frame)
    brt = br.transform_to(frame)
    tlt = tl.transform_to(frame)
    trt = tr.transform_to(frame)
    xys = [parent_ax.wcs.wcs_world2pix([[crd.spherical.lon.deg,
                                         crd.spherical.lat.deg]],0)[0]
           for crd in (blt, brt, trt, tlt, blt)]

    markinkwargs = dict(fc='none', ec=edgecolor)
    ppoly = Polygon(xys, fill=False, **markinkwargs)
    parent_ax.add_patch(ppoly)

    x1,y1,x2,y2 = axins.bbox.extents
    print(axins.bbox.extents)
    corners = parent_ax.transData.transform(xys)
    print(xys, corners)

    # p1 = PathPatch(Path([[x1, y1], corners[loc1]]),
    #                color=edgecolor)
    # axins.figure.add_patch(p1)
    # p1.set_clip_on(False)
    # p2 = PathPatch(Path([[x2, y2], corners[loc2]]),
    #                color=edgecolor)
    # axins.figure.add_patch(p2)
    # p2.set_clip_on(False)

    con1 = ConnectionPatch(xyA=(x1, y1), coordsA=axins.transData,
                           xyB=corners[loc1], coordsB=parent_ax.transData,
                           # linestyle='--', color='black', zorder=-1)
                           linestyle='--', color=edgecolor, zorder=-1)
    con2 = ConnectionPatch(xyA=x2, coordsA=axins.transData,
                           xyB=corners[loc2], coordsB=parent_ax.transData,
                           linestyle='--', color=edgecolor, zorder=-1)
    #axins.add_patch(con1)
    #axins.add_patch(con2)
    fig.add_patch(con1)
    fig.add_patch(con2)

def hide_ticks(ax):
    ra = ax.coords['ra']
    dec = ax.coords['dec']
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ra.set_ticks_visible(False)
    dec.set_ticks_visible(False)
    ra.set_axislabel('')
    dec.set_axislabel('')
    ra.ticklabels.set_visible(False)
    dec.ticklabels.set_visible(False)



def make_scalebar(ax, left_side, length, color='w', linestyle='-', label='',
                  fontsize=12, text_offset=0.1*u.arcsec):
    axlims = ax.axis()
    lines = ax.plot(u.Quantity([left_side.ra, left_side.ra-length]),
                    u.Quantity([left_side.dec]*2),
                    color=color, linestyle=linestyle, marker=None,
                    transform=ax.get_transform('fk5'),
                   )
    txt = ax.text((left_side.ra-length/2).to(u.deg).value,
                  (left_side.dec+text_offset).to(u.deg).value,
                  label,
                  verticalalignment='bottom',
                  horizontalalignment='center',
                  transform=ax.get_transform('fk5'),
                  color=color,
                  fontsize=fontsize,
                 )
    ax.axis(axlims)
    return lines,txt



def make_zoom(filename, zoom_parameters,
              overview_vis_pars={'max_percent':99.5, 'min_percent':0.5, 'stretch':'linear'},
              overview_cmap='gray_r',
              inset_cmap='inferno',
              main_zoombox=None,
              scalebar_loc=(0.1,0.1),
              scalebar_length=0.1*u.pc,
              beam_loc=(0.05, 0.05),
              nsigma_asinh=5,
              nsigma_max=10,
              nticks_inset=7,
              fontsize=20,
              tick_fontsize=16,
              distance=8.5*u.kpc
             ):

    fh = fits.open(filename)
    img = fh[0].data
    ww = WCS(fh[0].header)

    # cut out the minimal region that has data in it
    lbx,lby = np.argmax(np.any(notnan, axis=0)), np.argmax(np.any(notnan, axis=1))
    rtx,rty = notnan.shape[1] - np.argmax(np.any(notnan[::-1,::-1], axis=0)), notnan.shape[0] - np.argmax(np.any(notnan[::-1,::-1], axis=1))

    img = img[lby:rty, lbx:rtx]
    ww = ww[lby:rty, lbx:rtx]

    radesys = ww.wcs.radesys.upper()

    fig = pl.figure(1, figsize=(10,10))
    fig.clf()
    ax = fig.add_subplot(projection=ww.celestial)

    norm = simple_norm(img, **overview_vis_pars)

    img[img==0] = np.nan
    mad = mad_std(img, ignore_nan=True)

    if hasattr(norm.stretch, 'a') and nsigma_asinh is not None:
        norm.vmax = (np.nanmedian(img) + nsigma_max*mad)
        a_point = (np.nanmedian(img) + nsigma_asinh*mad) / norm.vmax
        norm.stretch.a = a_point
        print(f"numbers for norm: {np.nanmedian(img), nsigma_asinh, mad, nsigma_asinh*mad, norm.vmax, a_point}")

    im = ax.imshow(img, cmap=overview_cmap, norm=norm)

    ra = ax.coords['ra']
    ra.set_major_formatter('hh:mm:ss.s')
    dec = ax.coords['dec']
    ra.set_axislabel(f"RA ({radesys})", fontsize=fontsize)
    dec.set_axislabel(f"Dec ({radesys})", fontsize=fontsize, minpad=0.0)
    ra.ticklabels.set_fontsize(tick_fontsize)
    ra.set_ticks(exclude_overlapping=True)
    dec.ticklabels.set_fontsize(tick_fontsize)
    dec.set_ticks(exclude_overlapping=True)


    for zp in zoom_parameters:

        xl,xr = zp['xl'], zp['xr']
        yl,yu = zp['yl'], zp['yu']
        slc = [slice(yl,yu), slice(xl,xr)]
        axins = inset_axes(ax, **zp['inset_pars'],
                           axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                           axes_kwargs=dict(wcs=ww.celestial))

        norm2 = simple_norm(img, **zp['vis_pars'])

        inset_cm = copy.copy(pl.cm.get_cmap(inset_cmap))
        inset_cm.set_bad(inset_cm(0))

        im_ins = axins.imshow(img[slc], extent=[xl,xr,yl,yu], cmap=inset_cm, norm=norm2)
        mark_inset(parent_axes=ax, inset_axes=axins,
                   fc="none", ec="b", **zp['mark_inset_pars'])
        ra = axins.coords['ra']
        dec = axins.coords['dec']
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)
        ra.set_ticks_visible(False)
        dec.set_ticks_visible(False)
        ra.set_axislabel('')
        dec.set_axislabel('')
        ra.ticklabels.set_visible(False)
        dec.ticklabels.set_visible(False)

        caxins = inset_axes(axins,
                 width="5%", # width = 10% of parent_bbox width
                 height="100%", # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(1.05, 0., 1, 1),
                 bbox_transform=axins.transAxes,
                 borderpad=0,
                 )

        cbins = pl.colorbar(mappable=im_ins, cax=caxins)
        cbins.ax.tick_params(labelsize=tick_fontsize)
        cbins.set_label(f"S$_\\nu$ [Jy beam$^{-1}$]", fontsize=fontsize)

        if 'tick_locs' in zp:
            cbins.set_ticks(zp['tick_locs'])
            if 'tick_labels' in zp:
                cbins.set_ticklabels(zp['tick_labels'])
        elif 'asinh' in str(norm2.stretch).lower():
            rounded_loc, rounded = determine_asinh_ticklocs(norm2.vmin, norm2.vmax, nticks=nticks_inset)
            cbins.set_ticks(rounded_loc)
            cbins.set_ticklabels(rounded)
        elif'log' in str(norm2.stretch).lower():
            if norm2.vmin > 0:
                rounded_loc, rounded = determine_asinh_ticklocs(norm2.vmin, norm2.vmax, nticks=nticks_inset, rms=mad, stretch='log')
                cbins.set_ticks(rounded_loc)
                cbins.set_ticklabels(rounded)
            else:
                ticks = cbins.get_ticks()
                newticks = [norm2.vmin] + list(ticks)
                newticks = [norm2.vmin, 0,] + list(np.geomspace(mad, norm2.vmax, 4))
                print(f"ticks={ticks}, newticks={newticks}, mad={mad}, vmin={norm2.vmin}")
                cbins.set_ticks(newticks)

    if main_zoombox:
        ax.axis(main_zoombox)

    divider = make_axes_locatable(ax)
    cax1 = fig.add_axes([ax.get_position().x1+0.01,
                         ax.get_position().y0,
                         0.02,
                         ax.get_position().height])
    cb1 = pl.colorbar(mappable=im, cax=cax1)
    cb1.ax.tick_params(labelsize=tick_fontsize)
    cb1.set_label(f"S$_\\nu$ [Jy beam$^{-1}$]", fontsize=fontsize)
    pl.setp(cb1.ax.yaxis.get_label(), backgroundcolor="white")

    left_side = coordinates.SkyCoord(*ww.celestial.wcs_pix2world(scalebar_loc[1]*img.shape[1],
                                                                        scalebar_loc[0]*img.shape[0], 0)*u.deg, frame='fk5')
    length = (scalebar_length / distance).to(u.arcsec, u.dimensionless_angles())
    make_scalebar(ax, left_side, length, color='k', linestyle='-', label=f'{scalebar_length:0.1f}',
                  text_offset=0.5*u.arcsec, fontsize=fontsize)

    beam = radio_beam.Beam.from_fits_header(fh[0].header)
    ell = beam.ellipse_to_plot(beam_loc[1]*img.shape[1], beam_loc[0]*img.shape[0], pixscale=ww.celestial.pixel_scale_matrix[1,1]*u.deg)
    ax.add_patch(ell)
