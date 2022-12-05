import typing as tp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


from .sensor_client import GtacInterface


class _ArtistObjectHandlerYData:
    """Base class to handle data with only y axis."""

    def __init__(self, data: tp.Sequence[float], ax: plt.Axes):
        self._d = np.array(data)
        self._n = self._d.shape[0]
        (self._obj,) = ax.plot(np.zeros(self._n))

    def update(self, new_value):
        self._d[: self._n - 1] = self._d[1 : self._n]
        self._d[self._n - 1] = new_value
        self._obj.set_ydata(self._d)

    @property
    def artistic_object(self):
        return self._obj


class _NormalForce(_ArtistObjectHandlerYData):
    def __init__(self, data, ax):
        super(_NormalForce, self).__init__(data, ax)
        ax.set_ylim(0, 2500)


class GtacVisualizer:
    def __init__(self, interface: GtacInterface, figsize=None, num_data=500):
        figsize = (10, 10) if figsize is None else figsize
        self._gtac: GtacInterface = interface
        self._fig, _ = plt.subplots(1, 1, figsize=(10, 10))

        self._normal_force = _NormalForce(np.zeros(num_data), self._fig.axes[0])

        self._anim = None

    def _animate_fn(self, i):
        self._normal_force.update(self._gtac.forces[2])
        return (self._normal_force.artistic_object,)

    def show(self):
        self._gtac.zero()
        self._anim = FuncAnimation(
            self._fig, self._animate_fn, frames=2, interval=10, blit=True
        )
        plt.show()
