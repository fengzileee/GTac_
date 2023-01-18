import logging
import typing as tp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


from .sensor_client import GtacInterface


logger = logging.getLogger(__name__)


class _ArtistHandlerLineY:
    """Base class to handle time-varying data for line plot.

    Use init in the children to beautify the axis.
    """

    def __init__(self, data: tp.Sequence[float], ax: plt.Axes):
        self._d = np.array(data)
        self._n = self._d.shape[0]
        (self._obj,) = ax.plot(np.zeros(self._n))

    def update(self, new_value):
        self._d[: self._n - 1] = self._d[1 : self._n]
        self._d[self._n - 1] = new_value
        self._obj.set_ydata(self._d)

    @property
    def artist(self):
        return self._obj

    @property
    def data_len(self):
        return self._n


class _NormalForce(_ArtistHandlerLineY):
    def __init__(self, data, ax: plt.Axes):
        super(_NormalForce, self).__init__(data, ax)
        ax.set_ylim(-50, 2500)
        ax.set_xlim(0, self.data_len)
        ax.plot([0, self.data_len], [0, 0], "k--")


class _ShearForce(_ArtistHandlerLineY):
    def __init__(self, data, ax: plt.Axes):
        super(_ShearForce, self).__init__(data, ax)
        ax.set_ylim(-500, 500)
        ax.set_xlim(0, self.data_len)
        ax.plot([0, self.data_len], [0, 0], "k--")


class _ArtistHandlerScatter:
    """Base class to handle time-varying data for scatter plot.

    Use init in the children to beautify the axis.
    """

    def __init__(
        self,
        x_data: tp.Sequence[float],
        y_data: tp.Sequence[float],
        size_data: tp.Sequence[float],
        ax: plt.Axes,
    ):
        self._offset = np.array([x_data, y_data]).T
        self._size = size_data
        self._obj: matplotlib.collections.PathCollection = ax.scatter(
            np.array(x_data), np.array(y_data)
        )
        self._obj.set_sizes(self._size)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xlabel("sensor X axis")
        ax.set_ylabel("sensor Y axis")
        ax.set_aspect(1)

    def update(self, x=None, y=None, size=None):
        if x is not None:
            self._offset[:, 0] = x
        if y is not None:
            self._offset[:, 1] = y
        if size is not None:
            self._size = size
        self._obj.set_offsets(self._offset)
        self._obj.set_sizes(self._size)

    @property
    def artist(self):
        return self._obj


bubble_init_x = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
bubble_init_y = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]


class _Bubble(_ArtistHandlerScatter):
    def __init__(
        self,
        x_data: tp.Sequence[float],
        y_data: tp.Sequence[float],
        size_data: tp.Sequence[float],
        ax: plt.Axes,
    ):
        super(_Bubble, self).__init__(x_data, y_data, size_data, ax)
        ax.scatter(np.array(bubble_init_x), np.array(bubble_init_y), s=1500, alpha=0.4)


# Keys (and legend strings) of the artist handlers
# naming convention: starts with k
k_normal_f = "normal force"
k_normal_f_sum = "normal force 16 sum"
k_shear_f_x = "shear force x"
k_shear_f_y = "shear force y"
k_bubbles = "bubbles"


class GtacVisualizer:
    """Visualize the data of a GTac sensor."""

    def __init__(self, interface: GtacInterface, figsize=None, num_data=500):
        figsize = (10, 10) if figsize is None else figsize
        self._gtac: GtacInterface = interface
        self._fig, _ = plt.subplots(3, 1, figsize=(10, 10))

        self._h = {}
        # axis for normal force
        self._h[k_normal_f] = _NormalForce(np.zeros(num_data), self._fig.axes[0])
        self._h[k_normal_f_sum] = _NormalForce(np.zeros(num_data), self._fig.axes[0])
        self._add_legends_to_artists(self._fig.axes[0], [k_normal_f, k_normal_f_sum])
        # axis for shear forces
        self._h[k_shear_f_x] = _ShearForce(np.zeros(num_data), self._fig.axes[1])
        self._h[k_shear_f_y] = _ShearForce(np.zeros(num_data), self._fig.axes[1])
        self._add_legends_to_artists(self._fig.axes[1], [k_shear_f_x, k_shear_f_y])
        # axis for individual bubbles
        self._h[k_bubbles] = _Bubble(
            bubble_init_x, bubble_init_y, 100 * np.ones(16), self._fig.axes[2]
        )

        self.anim = None

    def _animate_fn(self, i):
        self._h[k_normal_f].update(self._gtac.forces[2])
        self._h[k_normal_f_sum].update(self._gtac.pressures.sum(axis=None))
        self._h[k_shear_f_x].update(self._gtac.forces[0])
        self._h[k_shear_f_y].update(self._gtac.forces[1])
        self._h[k_bubbles].update(size=np.abs(self._gtac.pressures.flatten()))
        return [v.artist for (_, v) in self._h.items()]

    def show(self):
        self._gtac.zero()
        self.anim = FuncAnimation(
            self._fig, self._animate_fn, frames=2, interval=10, blit=True
        )
        plt.show()

    def _add_legends_to_artists(self, ax, artists: tp.List[str]):
        """Add legends to artists in the same axes."""
        ax.legend(handles=[self._h[n].artist for n in artists], labels=artists)
