from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import ndarray

from CustomMath import normalizeArray


def getParameterInsetAxes(
    axes: Axes,
    parameter_values: Dict[str, Tuple[float, float]],
    color: str = "black",
    marker: str = 'o',
    markersize: float = 2,
    linestyle: str = '',
    clip_on: bool = False,
    labelpad: int = 2,
    fontsize: int = 8
) -> Tuple[Axes, partial]:
    """
    Get inset axes object and partially-completed plot for inset axes.
    This inset plot displays values for up to two free parameters during a parameter sweep.
    
    :param axes: axes to nest inset plot inside into
    :param parameter_values: dictionary of parameter values.
        Key is name of parameter.
        Value is 2-tuple of (minimum, maximum) values for parameter.
    :param color: marker color for inset plot
    :param marker: marker type for inset plot
    :param markersize: marker size inset plot
    :param linestyle: linestyle for inset plot
    :param clip_on: set True to allow points to extend outside inset plot. Set False otherwise.
    :param labelpad: padding for axis label(s)
    :param fontsize: fontsize for axis label(s)
    :returns: Tuple of (inset axes object, plot function taking single point of free parameters as input).
    """
    parameter_names = list(parameter_values.keys())
    parameter_count = len(parameter_names)
    axins_kwargs = {
        "color": color,
        "marker": marker,
        "markersize": markersize,
        "linestyle": linestyle,
        "clip_on": clip_on
    }
    label_kwargs = {
        "labelpad": labelpad,
        "fontsize": fontsize
    }

    location = [0.88, 0.88, 0.1, 0.1]  # [1, 1.035, 0.1, 0.1]
    axins = axes.inset_axes(location)
    axins.set_xticks(())
    axins.set_yticks(())
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    if parameter_count == 2:
        axins.set_ylabel(parameter_names[1], **label_kwargs)
        ylim = list(parameter_values.values())[1]
        axins_plot = partial(axins.plot, **axins_kwargs)
    elif parameter_count == 1:
        axins.spines["left"].set_visible(False)
        axins.spines["bottom"].set_position(("axes", 0.5))
        ylim = -1, 1
        axins_plot = lambda x: axins.plot(x, 0, **axins_kwargs)
    else:
        raise ValueError("must have exactly one or two parameters in inset plot")

    xlim = list(parameter_values.values())[0]
    axins.set_xlabel(parameter_names[0], **label_kwargs)
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.set_facecolor("none")

    return axins, axins_plot


def getLimits(
    data: ndarray, 
    limits: Tuple[Optional[float], Optional[float]],
    autoscale_on: bool=False,
    relative_padding: float=0.05
) -> Tuple[float, float]:
    """
    Get axis limits.

    :param data: data to retrieve limits for
    :param limits: (lower, upper) limits for axis range.
        Default roughly to (minimum, maximum) for each value not given.
        Ignored when :paramref:`~CustomFigure.getFigure.getLimits.autoscale_on` is True.
    :param autoscale_on: whether or not to autoscale data from its minimum and maximum.
        Defaults to False.
        Overrides :paramref:`~CustomFigure.getFigure.getLimits.limits` when True.
    :param padding: proportion of space (out of full range) to leave between data and line.
        Only called if at least one limit is not given.
    :returns: Tuple of (minimum, maximum) range for plot axis.
        Defaults to (data.min(), data.max()) with proportional padding
    """
    min_limit, max_limit = limits
    
    exist_min_limit = isinstance(min_limit, float)
    exist_max_limit = isinstance(max_limit, float)
    
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    absolute_padding = relative_padding * data_range
    
    scaled_limits = [
        data_min - absolute_padding, 
        data_max + absolute_padding
    ]
    if not autoscale_on:
        if exist_min_limit:
            scaled_limits[0] = min_limit
        if exist_max_limit:
            scaled_limits[1] = max_limit
    
    scaled_limits = tuple(scaled_limits)
    return scaled_limits


def getFigure(
    results: Dict[str, ndarray],
    scale_factor: Dict[str, float] = None,
    normalize: Dict[str, bool] = None,
    plot_type: str = "xy",
    segment_count: int = 100,
    clim: Tuple[Optional[float], Optional[float]] = (None, None),
    autoscalec_on: bool = True,
    colormap: str = None,
    colorbar_kwargs: Dict[str, Any] = None,
    axes_kwargs: Dict[str, Any] = None,
    plot_kwargs: Dict[str, Any] = None,
    inset_parameters: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = None
) -> Figure:
    """
    Get matplotlib figure from data.

    :param results: dictionary of results.
        Key is name of axis.
        Value is values to to plot along axis.
    :param scale_factor: dictionary of scale factor for each axis.
        Key is name of axis.
        Value is scale factor.
        Result along corresponding axis is divided by this factor.
        Defaults to 1 for each axis not given.
        Overriden by :paramref:`~SimulationWindow.getFigure.normalize` if set to True.
    :param normalize: dictionary of boolean indicating whether to normalize each axis.
        Key is name of axis.
        Set True to normalize axis values by maximum absolute value.
        Set False to retain original scaling.
        Defaults to False.
    :param clim: axis bounds for colorbar.
        First element gives lower limit.
        Defaults to minium of :paramref:`~SimulationWindow.getFigure.c`.
        Second element gives upper limit.
        Defaults to maximum of :paramref:`~SimulationWindow.getFigure.c`.
    :param autoscalec_on: set True to autoscale colorbar axis.
        Set False otherwise.
        Overrides :paramref:`~SimulationWindow.getFigure.clim` when set True.
    :param colormap: colormap to use for colorbar.
    :param plot_type: type of plot to display.
        This could include a single curve, multiple curves, scatter plot.
    :param segment_count: number of segments for colorbar.
        Only called when a single line is multicolored.
    :param inset_parameters: 2-level dictionary of parameters to show on inset plot.
        Key is name of parameter.
        Subkeys are "range" and "value"
            "range": value is tuple (minimum, maximum) for parameter
            "value": value is value for parameter within range
    :param colorbar_kwargs: additional arguments to pass into :class:`matplotlib.pyplot.colorbar`
    :param axes_kwargs: additional arguments to pass into :class:`matplotlib.axes.Axes'
    :param plot_kwargs: additional arguments to pass into axes plot method
    """

    def coordinates2lines(
        x: ndarray, 
        y: ndarray, 
        z: ndarray=None, 
        c: ndarray=None, 
        cmap: str=None, 
        norm=None,
        segment_count: int=None
    ) -> LineCollection:
        """
        Convert coordinates from x, y(, z) into collection of line segment counts for plotting.
        x, y, z, c must be ordered to correspond to each other.
        
        :param x: x values of coordinates, 1D numpy array
        :param y: y values of coordinates, 1D numpy array
        :param z: z values of coordinates, 1D numpy array.
            Defaults to 2D-(xy-)plane if None.
        :param c: colors of coordinates, 1D numpy array
        :param cmap: name of colormap corresponding to :paramref:`~Layout.SimulationWindow.getFigure.coordinates2lines.c`
        :param norm: normalization of colormap corresponding to :paramref:`~Layout.SimulationWindow.getFigure.coordinates2lines.c`
        :param segment_count: number of line segments to interpolate from data.
            Defaults to all segments if None.
        """
        for array in [x, y, z, c]:
            if isinstance(array, ndarray):
                assert array.ndim == 1
        
        x_interpolate, y_interpolate = interpolateLines((x, y), segment_count)    
        if isinstance(z, ndarray):
            z_interpolate = interpolateLines(z, segment_count)
            points = np.array([x_interpolate, y_interpolate, z_interpolate]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line_collection = Line3DCollection(segments, cmap=cmap, norm=norm)
        elif z is None:
            points = np.array([x_interpolate, y_interpolate]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line_collection = LineCollection(segments, cmap=cmap, norm=norm)
        
        if isinstance(c, ndarray):
            c_interpolate = interpolateLines(c, segment_count)
            line_collection.set_array(c_interpolate)
        
        return line_collection
    
    def interpolateLines(xs, segment_count=None) -> Union[ndarray, Tuple[ndarray]]:
        """
        Get subarray(s) of given array(s).
        
        :param xs: (tuple of) array(s) to interpolate
        :param segment_count: number of segments (>0) in resultant subarray.
            Defaults to entire array if None.
        :returns: ndarray if xs is ndarray.
            Tuple of ndarray if xs is tuple of ndarray.
        """
        if segment_count is None:
            return xs
        assert isinstance(segment_count, int)
        assert segment_count >= 0
        
        if isinstance(xs, ndarray):
            xsize = xs.shape[0]
            if segment_count >= xsize:
                return xs
            elif segment_count < xsize:
                index_linspace = np.linspace(0, xsize-1, segment_count+1, dtype=np.int32)
                index_interpolate = np.round(index_linspace)
                x_interpolate = xs[index_interpolate]
                return x_interpolate
        elif isinstance(xs, tuple):
            xsize = xs[0].size
            for x in xs:
                assert isinstance(x, ndarray)
                assert x.ndim == 1
                assert x.size == xsize
            
            interpolateLinesPartial = partial(interpolateLines, segment_count=segment_count)
            xs_interpolate = tuple(list(map(interpolateLinesPartial, xs)))
            
            return xs_interpolate
    
    if plot_kwargs is None:
        plot_kwargs = {}
    if axes_kwargs is None:
        axes_kwargs = {}
    if scale_factor is None:
        scale_factor = {}
    if segment_count is not None:
        assert isinstance(segment_count, int)
        assert segment_count >= 0
    if normalize is None:
        normalize = {}
    
    axis_names = results.keys()
    for axis_name in axis_names:
        if axis_name not in normalize.keys():
            normalize[axis_name] = False
        elif normalize[axis_name]:
            abs_max = np.max(np.abs(results[axis_name]))
            scale_factor[axis_name] = abs_max
        elif axis_name not in scale_factor.keys():
            scale_factor[axis_name] = 1

    figure = plt.figure()

    if 'x' in plot_type:
        if normalize['x']:
            x = np.apply_along_axis(normalizeArray, -1, results['x'])
        else:
            x = results['x'] / scale_factor['x']
            
        xsize = x.size
        xshape = x.shape
        
        axes_kwargs["xlim"] = getLimits(
            x, 
            limits=axes_kwargs['xlim'], 
            autoscale_on=axes_kwargs["autoscalex_on"]
        )

    if 'y' in plot_type:
        if normalize['y']:
            y = np.apply_along_axis(normalizeArray, -1, results['y'])
        else:
            y = results['y'] / scale_factor['y']
        
        ysize = y.size
        yshape = y.shape
        axes_kwargs["ylim"] = getLimits(
            y, 
            limits=axes_kwargs['ylim'], 
            autoscale_on=axes_kwargs["autoscaley_on"]
        )

    if 'z' in plot_type:
        if normalize['z']:
            z = np.apply_along_axis(normalizeArray, -1, results['z'])
        else:
            z = results['z'] / scale_factor['z']
        
        zsize = z.size
        zshape = z.shape
        axes_kwargs["zlim"] = getLimits(
            z, 
            limits=axes_kwargs['zlim'], 
            autoscale_on=axes_kwargs["autoscalez_on"]
        )
        axes = figure.add_subplot(projection="3d")
    else:
        axes = figure.add_subplot()

    for axes_arg, axes_value in axes_kwargs.items():
        axes_kwarg = {axes_arg: axes_value}
        try:
            axes.set(**axes_kwarg)
        except AttributeError:
            print("axes_kwarg:", axes_kwarg)
    
    if 'c' in plot_type or 't' in plot_type:
        if colormap is None:
            colormap = "viridis"
        if colorbar_kwargs is None:
            colorbar_kwargs = {}
        
        if normalize['c']:
            if 't' in plot_type:
                c = np.apply_along_axis(normalizeArray, -1, results['c'])
            elif 'c' in plot_type:
                c = normalizeArray(results['c'])
        else:
            c = results['c'] / scale_factor['c']
            
        csize = c.size
        cshape = c.shape

        vmin, vmax = getLimits(
            c, 
            limits=clim, 
            autoscale_on=autoscalec_on
        )

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # noinspection PyUnresolvedReferences
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        figure.colorbar(cmap, **colorbar_kwargs)
        try:
            c_colors = cmap.to_rgba(c)
        except ValueError:
            pass

    if inset_parameters is not None:
        free_parameter_ranges = {name: inset_parameters[name]["range"] for name in inset_parameters.keys()}
        inset_axes, inset_axes_plot = getParameterInsetAxes(axes, free_parameter_ranges)
        free_parameter_values = tuple(inset_parameters[name]["value"] for name in inset_parameters.keys())
        inset_axes_plot(*free_parameter_values)
    
    if 'z' not in plot_type:
        if plot_type in ["xy", "nxy", "xny"]:
            assert xshape == yshape
            axes.plot(x, y, **plot_kwargs)
        elif plot_type == "cnxny":
            assert cshape == (xsize, ysize)
            c_shaped = c.T
            axes.contourf(
                x, y, c_shaped, 
                levels=segment_count, 
                cmap=colormap, 
                norm=norm, 
                **plot_kwargs
            )
        elif plot_type in ["ntxy", "cnxy", "cxny"]:
            assert cshape == xshape == yshape
            
            if segment_count == 0:
                axes.scatter(x, y, color=c_colors)
            elif segment_count > 0:
                line_collection = coordinates2lines(
                    x, y, c=c,
                    cmap=colormap,
                    norm=norm,
                    segment_count=segment_count
                )
                axes.add_collection(line_collection, **plot_kwargs)
        elif plot_type in ["ncxy", "ncxny", "ncnxy"]:
            if plot_type == "ncxy":
                assert xshape == yshape and xshape[0] == csize
                x_lines, y_lines = x, y
            elif plot_type == "ncxny":
                assert xshape == (csize, ysize)
                x_lines = x
                y_lines = np.tile(y, (csize, 1))
            elif plot_type == "ncnxy":
                assert yshape == (csize, xsize)
                x_lines = np.tile(x, (csize, 1))
                y_lines = y

            for line_index in range(csize):
                axes.plot(
                    x_lines[line_index],
                    y_lines[line_index], 
                    color=c_colors[line_index], 
                    **plot_kwargs
                )
        elif plot_type in ["ntnxy", "ntxny"]:
            if plot_type == "ntxny":
                assert cshape == xshape and cshape[0] == ysize
                x_lines = x
                y_lines = np.tile(y, (cshape[-1], 1)).T
                line_count = ysize
            elif plot_type == "ntnxy":
                assert cshape == yshape and cshape[0] == xsize
                x_lines = np.tile(x, (cshape[-1], 1)).T
                y_lines = y
                line_count = xsize

            for line_index in range(line_count):
                line_collection = coordinates2lines(
                    x_lines[line_index], 
                    y_lines[line_index], 
                    c=c[line_index], 
                    cmap=colormap, 
                    norm=norm,
                    segment_count=segment_count
                )
                axes.add_collection(line_collection, **plot_kwargs)
    elif 'z' in plot_type:
        if plot_type == "xyz":
            assert xshape == yshape == zshape
            axes.plot3D(x, y, z, **plot_kwargs)
        elif plot_type == "ntxyz":
            assert cshape == xshape == yshape == zshape

            if segment_count == 0:
                axes.scatter3D(x, y, z, color=c_colors)
            else:
                line_collection = coordinates2lines(
                    x, 
                    y, 
                    z, 
                    c=c,
                    cmap=colormap,
                    norm=norm,
                    segment_count=segment_count
                )
                axes.add_collection(line_collection, **plot_kwargs)
        elif plot_type in ["nxyz", "xnyz", "xynz"]:
            if plot_type == "nxyz":
                assert yshape == zshape and yshape[0] == xsize
                x_rots = np.tile(x, (yshape[1], 1)).T
                y_rots = y
                z_lines = z
                rot_size = xsize
            elif plot_type == "xnyz":
                assert xshape == zshape and xshape[0] == ysize
                x_rots = x
                y_rots = np.tile(y, (xshape[1], 1)).T
                z_lines = z
                rot_size = ysize
            elif plot_type == "xynz":
                assert xshape == yshape and xshape[0] == zsize
                x_rots = x
                y_rots = y
                z_lines = np.tile(z, (xshape[1], 1)).T
                rot_size = zsize

            for line_index in range(rot_size):
                axes.plot3D(
                    x_rots[line_index], 
                    y_rots[line_index], 
                    z_lines[line_index], 
                    **plot_kwargs
                )
        elif plot_type in ["ntnxyz", "ntxnyz", "ntxynz"]:
            if plot_type == "ntnxyz":
                assert cshape == yshape == zshape and yshape[0] == xsize
                x_rots, y_rots, zs = y, z, x
                zdir = 'x'
            elif plot_type == "ntxnyz":
                assert cshape == xshape == zshape and xshape[0] == ysize
                x_rots, y_rots, zs = x, z, y
                zdir = 'y'
            elif plot_type == "ntxynz":
                assert cshape == xshape == yshape and xshape[0] == zsize
                x_rots, y_rots, zs = x, y, z
                zdir = 'z'

            for line_index in range(zs.size):
                line_collection = coordinates2lines(
                    x_rots[line_index], 
                    y_rots[line_index], 
                    c=c[line_index], 
                    cmap=colormap, 
                    norm=norm,
                    segment_count=segment_count
                )
                axes.add_collection3d(
                    line_collection, 
                    zs=zs[line_index], 
                    zdir=zdir, 
                    **plot_kwargs
                )
        elif plot_type in ["ncxyz", "ncnxyz", "ncxnyz", "ncxynz"]:
            if plot_type == "ncxyz":
                assert xshape == yshape == zshape and xshape[0] == csize
                x_rots, y_rots, z_lines = x, y, z
                extra_dimension = False
            elif plot_type == "ncnxyz":
                assert yshape[0:2] == (csize, xsize) and yshape == zshape
                try:
                    x_rots = np.transpose(np.tile(x, (csize, yshape[2], 1)), axes=(0, 2, 1))
                    rot_size = xsize
                    extra_dimension = True
                except IndexError:
                    x_rots = np.tile(x, (csize, 1))
                    extra_dimension = False
                y_rots, z_lines = y, z
            elif plot_type == "ncxnyz":
                assert xshape[0:2] == (csize, ysize) and xshape == zshape
                x_rots, z_lines = x, z
                try:
                    y_rots = np.transpose(np.tile(y, (csize, xshape[2], 1)), axes=(0, 2, 1))
                    rot_size = ysize
                    extra_dimension = True
                except IndexError:
                    y_rots = np.tile(y, (csize, 1))
                    extra_dimension = False
            elif plot_type == "ncxynz":
                assert xshape[0:2] == (csize, zsize) and xshape == yshape
                x_rots, y_rots = x, y
                try:
                    z_lines = np.transpose(np.tile(z, (csize, xshape[2], 1)), axes=(0, 2, 1))
                    rot_size = zsize
                    extra_dimension = True
                except IndexError:
                    z_lines = np.tile(z, (csize, 1))
                    extra_dimension = False

            for c_index in range(csize):
                c_color = c_colors[c_index]
                if extra_dimension:
                    for z_index in range(rot_size):
                        line_index = (c_index, z_index)
                        axes.plot3D(
                            x_rots[line_index],
                            y_rots[line_index],
                            z_lines[line_index],
                            color=c_color,
                            **plot_kwargs
                        )
                else:
                    axes.plot3D(
                        x_rots[c_index],
                        y_rots[c_index],
                        z_lines[c_index],
                        color=c_color,
                        **plot_kwargs
                    )
        elif plot_type in ["ntnxnyz", "ntnxynz", "ntxnynz"]:
            if plot_type == "ntnxnyz":
                assert cshape[0:2] == (xsize, ysize) and cshape == zshape
                yrot_size = xsize
                x_rots = z
                y_rots = np.tile(x, (cshape[2], 1)).T
                zs, zdir = y, 'y'
                c_rots = c
            elif plot_type == "ntnxynz":
                assert cshape[0:2] == (xsize, zsize) and cshape == yshape
                yrot_size = zsize
                x_rots = np.transpose(y, axes=(1, 0, 2))
                y_rots = np.tile(z, (cshape[2], 1)).T
                zs, zdir = x, 'x'
                c_rots = np.transpose(c, axes=(1, 0, 2))
            elif plot_type == "ntxnynz":
                assert cshape[0:2] == (ysize, zsize) and cshape == xshape
                yrot_size = ysize
                x_rots = x
                y_rots = np.tile(y, (cshape[2], 1)).T
                zs, zdir = z, 'z'
                c_rots = c

            for y_index in range(yrot_size):
                for z_index in range(zs.size):
                    if plot_type == "ntnxnyz":
                        x_shaped = y_rots[y_index]
                        y_shaped = x_rots[y_index, z_index]
                    elif plot_type in ["ntnxynz", "ntxnynz"]:
                        x_shaped = x_rots[y_index, z_index]
                        y_shaped = y_rots[y_index]
                    c_shaped = c_rots[y_index, z_index]
                    
                    line_collection = coordinates2lines(
                        x_shaped,
                        y_shaped,
                        c=c_shaped,
                        cmap=colormap,
                        norm=norm,
                        segment_count=segment_count
                    )
                    axes.add_collection3d(
                        line_collection, 
                        zs=zs[z_index], 
                        zdir=zdir, 
                        **plot_kwargs
                    )
        elif plot_type in ["nxnyz", "tnxnyz"]:
            x_shaped, y_shaped = np.meshgrid(x, y)
            x_shaped = x_shaped.T
            y_shaped = y_shaped.T
            
            if plot_type == "nxnyz":
                axes.plot_surface(x_shaped, y_shaped, z)
            elif plot_type == "tnxnyz":
                axes.plot_surface(
                    x_shaped, 
                    y_shaped, 
                    z,
                    facecolors=c_colors
                )
        elif plot_type in ["cnxnyz", "cnxynz", "cxnynz", "cnxnynz"]:
            if plot_type == "cnxnyz":
                assert cshape == zshape == (xsize, ysize)
                c_shaped = c.reshape(csize)
                x_shaped = np.tile(x, (ysize, 1)).T
                y_shaped = np.tile(y, (xsize, 1))
                z_shaped = z
            elif plot_type == "cnxynz":
                assert cshape == yshape == (xsize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = np.tile(x, (zsize, 1)).T
                y_shaped = y
                z_shaped = np.tile(z, (xsize, 1))
            elif plot_type == "cxnynz":
                assert cshape == xshape == (ysize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = x
                y_shaped = np.transpose(np.tile(y, (zsize, 1)))
                z_shaped = np.tile(z, (ysize, 1))
            elif plot_type == "cnxnynz":
                assert cshape == (xsize, ysize, zsize)
                c_shaped = c.reshape(csize)
                x_shaped = np.transpose(np.tile(x, (ysize, zsize, 1)), axes=(2, 0, 1))
                y_shaped = np.transpose(np.tile(y, (zsize, xsize, 1)), axes=(1, 2, 0))
                z_shaped = np.tile(z, (xsize, ysize, 1))

            axes.scatter3D(
                x_shaped, 
                y_shaped, 
                z_shaped, 
                color=cmap.to_rgba(c_shaped)
            )
        elif plot_type in ["ncnxnyz", "ncnxynz", "ncxnynz"]:
            if plot_type == "ncnxnyz":
                assert zshape == (csize, xsize, ysize)
                x_shaped = np.transpose(np.tile(x, (ysize, csize, 1)), axes=(1, 2, 0))
                y_shaped = np.tile(y, (csize, xsize, 1))
                z_shaped = z
            elif plot_type == "ncnxynz":
                assert yshape == (csize, xsize, zsize)
                x_shaped = np.transpose(np.tile(x, (zsize, csize, 1)), axes=(1, 2, 0))
                y_shaped = y
                z_shaped = np.tile(z, (csize, xsize, 1))
            elif plot_type == "ncxnynz":
                assert xshape == (csize, ysize, zsize)
                x_shaped = x
                y_shaped = np.transpose(np.tile(y, (zsize, csize, 1)), axes=(1, 2, 0))
                z_shaped = np.tile(z, (csize, ysize, 1))

            for c_index in range(csize):
                axes.scatter3D(
                    x_shaped[c_index],
                    y_shaped[c_index],
                    z_shaped[c_index],
                    color=c_colors[c_index]
                )
                """axes.plot_wireframe(
                    x_shaped[c_index],
                    y_shaped[c_index],
                    z[c_index],
                    color=c_colors[c_index]
                )"""
            
        axes.set_xlim3d(*axes_kwargs["xlim"])
        axes.set_ylim3d(*axes_kwargs["ylim"])
        axes.set_zlim3d(*axes_kwargs["zlim"])
        axes.set_zlabel(axes_kwargs["zlabel"])

    return figure
