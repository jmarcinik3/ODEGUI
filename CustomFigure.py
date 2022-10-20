from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import ndarray
from scipy import interpolate

from Transforms.CustomMath import normalizeArray

axis_names = ('x', 'y', 'z', 'c', 'C', 'X', 'Y')


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
        def axins_plot(x): return axins.plot(x, 0, **axins_kwargs)
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
    autoscale_on: bool = False,
    relative_padding: float = 0.05
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
    if data is None:
        return (None, None)
    else:
        assert isinstance(data, ndarray)

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


def plotOnAxes(
    axes: Axes,
    x: ndarray,
    y: ndarray,
    z: ndarray = None,
    c: ndarray = None,
    C: ndarray = None,
    plot_type: str = "xy",
    clim: Tuple[float, float] = None,
    colormap: str = None,
    plot_kwargs: Dict = None,
    segment_count: int = 0,
    contour_levels: List[float] = None
):
    """
    Plot data onto single Axes object of figure.

    :param axes: Axes object to plot data on
    :param x: data for x-axis of plot
    :param y: data for y-axis of plot
    :param z: data for z-axis of plot
    :param c: data for color-axis of plot
    :param C: data for contour-axis of plot
    :param plot_type: type of plot to display.
        This could include a single curve, multiple curves, scatter plot.
    :param clim: limits for colorbar scheme of plot.
        Only called if plot type requires color axis.
    :param colormap: name of colormap for data.
        Only called if plot type requires color axis.
    :param plot_kwargs: additional arguments to pass into matplotlib.pyplot.plot or equivalent
    :param segment_count: number of segments to plot for multicolored-line(s) plot.
        Number of contour lines to plot for contour-field plot.
        Set to zero to substitute scatter plot in for standard-line plot.
        Only called when relevant to plot type.
    :param contour_levels: collection of values to plot contour lines at.
        Only called if plot type requires contour axis.
    """
    def coordinates2lines(
        x: ndarray,
        y: ndarray,
        z: ndarray = None,
        c: ndarray = None,
        norm=None,
        cmap: Union[str, cm.ScalarMappable] = None,
        segment_count: int = None
    ) -> LineCollection:
        """
        Convert coordinates from x, y(, z) into collection of line segment counts for plotting.
        x, y, z, c must be ordered to correspond to each other.

        :param x: x values of coordinates, 1D numpy array
        :param y: y values of coordinates, 1D numpy array
        :param z: z values of coordinates, 1D numpy array.
            Defaults to 2D-(xy-)plane if None.
        :param c: colors of coordinates, 1D numpy array
        :param cmap: (name of) colormap corresponding to :paramref:`~Layout.SimulationWindow.getFigure.coordinates2lines.c`
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
                index_linspace = np.linspace(0, xsize - 1, segment_count + 1, dtype=np.int32)
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

    try:
        xsize, xshape = x.size, x.shape
        ysize, yshape = y.size, y.shape
    except AttributeError:
        return axes

    if 'z' in plot_type:
        assert isinstance(z, ndarray)
        try:
            zsize, zshape = z.size, z.shape
        except AttributeError:
            return axes

    if 'c' in plot_type or 't' in plot_type:
        assert isinstance(c, ndarray)
        assert isinstance(clim[0], float) and isinstance(clim[1], float)
        assert colormap is None or isinstance(colormap, str)
        try:
            csize, cshape = c.size, c.shape
        except AttributeError:
            return axes

        vmin, vmax = clim
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        try:
            c_colors = cmap.to_rgba(c)
        except ValueError:
            pass

    if 'C' in plot_type:
        contour_levels = [0.5]
        assert isinstance(C, ndarray)
        assert isinstance(contour_levels, list)
        try:
            Csize, Cshape = C.size, C.shape
        except AttributeError:
            return axes

    if 'z' not in plot_type:
        if plot_type in ["xy", "nxy", "xny"]:
            assert xshape == yshape
            axes.plot(x, y, **plot_kwargs)
        elif plot_type == "cnxny":
            assert xsize >= 2 and ysize >= 2
            if cshape == (xsize, ysize):
                c_shaped = c.T
                x_shaped = x
                y_shaped = y
            elif csize == xsize == ysize:
                x_shaped = np.unique(x)
                y_shaped = np.unique(y)
                points = np.array((x, y)).T
                X = np.array([*np.meshgrid(x_shaped, y_shaped)]).T
                c_shaped = interpolate.griddata(points, c, X, method="linear").T
            else:
                raise AssertionError

            axes.contourf(
                x_shaped, y_shaped, c_shaped,
                levels=segment_count,
                cmap=colormap,
                norm=norm,
                **plot_kwargs
            )
        elif plot_type == "nxnyC":
            assert Cshape == (xsize, ysize)
            C_shaped = C.T

            axes.contour(
                x, y, C_shaped,
                levels=contour_levels,
                **plot_kwargs
            )
        elif plot_type == "ncnxnyC":
            assert Cshape == (csize, xsize, ysize)
            C_shaped = np.transpose(C, axes=[0, 2, 1])

            for c_index in range(csize):
                axes.contour(
                    x,
                    y,
                    C_shaped[c_index],
                    levels=contour_levels,
                    colors=(c_colors[c_index], ),
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

    return axes


def getFigure(
    results: Dict[str, ndarray],
    plot_type: str = "xy",
    scale_factor: Dict[str, float] = None,
    normalize: Dict[str, bool] = None,
    segment_count: int = 100,
    colormap: str = None,
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
    :param axes_kwargs: axis-based arguments; additional arguments to pass into :class:`matplotlib.axes.Axes'.
        Supports lim for x,y,z,c,X,Y; use "{axis_name}lim".
        Supports autoscale_on for relevant axes; use "autoscale{axis_name}_on".
        Supports label for x,y,z,c,X,Y; use "{axis_name}label".
        Supports scale type for x,y,z; use "{axis_name}scale".
        Support colorbar location (see matplotlib Colorbar); use "cloc".
        Use "title" for plot title.
    :param plot_kwargs: additional arguments to pass into axes plot method
    """

    if plot_kwargs is None:
        plot_kwargs = {}
    if axes_kwargs is None:
        axes_kwargs = {}
    if scale_factor is None:
        scale_factor = {}
    if normalize is None:
        normalize = {}

    assert isinstance(plot_kwargs, dict)
    assert isinstance(axes_kwargs, dict)
    assert isinstance(scale_factor, dict)
    assert isinstance(segment_count, int) and segment_count >= 0
    assert isinstance(normalize, dict)

    result_axis_names = results.keys()
    for axis_name in axis_names:
        if axis_name not in normalize.keys():
            normalize[axis_name] = False
        if axis_name not in scale_factor.keys():
            scale_factor[axis_name] = 1

    def getScaledResults(axis_name):
        result = np.nan_to_num(results[axis_name])

        if normalize[axis_name]:
            x = np.apply_along_axis(normalizeArray, -1, result)
        else:
            x = None if result is None else result / scale_factor[axis_name]

        return x

    for axis_name in axis_names:
        results[axis_name] = getScaledResults(axis_name) if axis_name in result_axis_names else None

    for axis_name in result_axis_names:
        lim_str = f"{axis_name:s}lim"
        try:
            autoscale_on = axes_kwargs[f"autoscale{axis_name:s}_on"]
        except KeyError:
            autoscale_on = False
            axes_kwargs[f"autoscale{axis_name:s}_on"] = autoscale_on

        try:
            axes_kwargs[lim_str]
        except KeyError:
            axes_kwargs[lim_str] = (None, None)
        given_limits = axes_kwargs[lim_str]

        axes_kwargs[lim_str] = getLimits(
            results[axis_name],
            limits=given_limits,
            autoscale_on=autoscale_on
        )

    def getSubsetResults(axis_name, x_index=None, y_index=None):
        result = results[axis_name]

        if result is None:
            pass
        elif len(result.shape) > 1:
            if 'X' in plot_type:
                result = result[x_index]
            if 'Y' in plot_type:
                result = result[y_index]

        return result

    def setAxesKwargs(axes):
        if 'z' in plot_type:
            axes.set_xlim3d(*axes_kwargs["xlim"])
            axes.set_ylim3d(*axes_kwargs["ylim"])
            axes.set_zlim3d(*axes_kwargs["zlim"])
            axes.set_zlabel(axes_kwargs["zlabel"])
            axes.set_zscale(axes_kwargs["zscale"])
            axes.set_autoscalez_on(axes_kwargs["autoscalez_on"])
        elif 'z' not in plot_type:
            try:
                axes.set_xlim(*axes_kwargs["xlim"])
            except (TypeError, KeyError):
                axes.set_xlim((None, None))
            try:
                axes.set_ylim(*axes_kwargs["ylim"])
            except (TypeError, KeyError):
                axes.set_ylim((None, None))

        try:
            axes.set_xlabel(axes_kwargs["xlabel"])
        except KeyError:
            pass
        try:
            axes.set_ylabel(axes_kwargs["ylabel"])
        except KeyError:
            pass
        try:
            axes.set_xscale(axes_kwargs["xscale"])
        except KeyError:
            pass
        try:
            axes.set_yscale(axes_kwargs["yscale"])
        except KeyError:
            pass
        try:
            axes.set_autoscalex_on(axes_kwargs["autoscalex_on"])
        except KeyError:
            pass
        try:
            axes.set_autoscaley_on(axes_kwargs["autoscaley_on"])
        except KeyError:
            pass

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    figure = plt.figure()

    if 'X' in plot_type:
        X = getScaledResults('X')
        Xsize = X.size
    elif 'X' not in plot_type:
        Xsize = 1

    if 'Y' in plot_type:
        Y = getScaledResults('Y')
        Ysize = Y.size
    elif 'Y' not in plot_type:
        Ysize = 1

    if 'z' in plot_type:
        projection = "3d"
    elif 'z' not in plot_type:
        projection = None

    full_figure_axis = figure.add_subplot(
        Ysize,
        Xsize,
        (1, Xsize * Ysize),
        frameon=False
    )
    full_figure_axis.xaxis.set_ticks([])
    full_figure_axis.yaxis.set_ticks([])

    try:
        full_figure_axis.set_title(axes_kwargs["title"])
    except KeyError:
        pass
    try:
        full_figure_axis.set_xlabel(axes_kwargs["Xlabel"])
    except KeyError:
        pass
    try:
        full_figure_axis.set_ylabel(axes_kwargs["Ylabel"])
    except KeyError:
        pass

    if inset_parameters is not None:
        free_parameter_ranges = {name: inset_parameters[name]["range"] for name in inset_parameters.keys()}
        inset_axes, inset_axes_plot = getParameterInsetAxes(full_figure_axis, free_parameter_ranges)
        free_parameter_values = tuple(inset_parameters[name]["value"] for name in inset_parameters.keys())
        inset_axes_plot(*free_parameter_values)

    axes = np.zeros((Xsize, Ysize), dtype=object)

    for x_index in range(Xsize):
        for y_index in range(Ysize):
            if x_index == 0 and y_index == 0:
                sharex = None
                sharey = None
            else:
                sharex = axes[0, 0]
                sharey = axes[0, 0]

            axis = figure.add_subplot(
                Ysize,
                Xsize,
                Xsize * (Ysize - y_index - 1) + x_index + 1,
                projection=projection,
                sharex=sharex,
                sharey=sharey
            )

            axis_visible = (x_index == 0) & (y_index == 0)
            axis.yaxis.set_visible(axis_visible)
            axis.xaxis.set_visible(axis_visible)
            axes[x_index, y_index] = axis

    setAxesKwargs(axes[0, 0])

    if 'c' in plot_type or 't' in plot_type:
        assert colormap is None or isinstance(colormap, str)

        vmin, vmax = axes_kwargs["clim"]
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.ScalarMappable(norm=cnorm, cmap=colormap)

        try:
            clabel = axes_kwargs["clabel"]
        except KeyError:
            clabel = ''
        try:
            cloc = axes_kwargs["cloc"]
        except KeyError:
            cloc = None

        figure.colorbar(
            mappable=cmap,
            ax=axes.ravel().tolist(),
            label=clabel,
            location=cloc
        )

    sub_plot_type = plot_type.split('_')[0]
    for x_index in range(Xsize):
        for y_index in range(Ysize):
            axis = axes[x_index, y_index]
            plotOnAxes(
                axis,
                x=getSubsetResults('x', x_index, y_index),
                y=getSubsetResults('y', x_index, y_index),
                z=getSubsetResults('z', x_index, y_index),
                c=getSubsetResults('c', x_index, y_index),
                C=getSubsetResults('C', x_index, y_index),
                plot_type=sub_plot_type,
                clim=axes_kwargs["clim"],
                colormap=colormap,
                segment_count=segment_count,
                plot_kwargs=plot_kwargs
            )

    return figure
