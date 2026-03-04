import numpy as np
from .package import PakBase


class Hydraulics(PakBase):
    """
    Package to define the Hydraulic parameters of a GroundwaterFlow model

    Parameters
    ----------

    """
    def __init__(self, parent, hk, vk, package_name="hyd"):
        super().__init__(parent, package_name)
        self._hk = np.ravel(hk)
        self._vk = np.ravel(vk)

    @property
    def hk(self):
        """
        Returns the horizontal hydraulic conductivity
        """
        return self._hk

    @property
    def vk(self):
        """
        Returns the vertical hydraulic conductivity
        """
        return self._vk

    @staticmethod
    def package_type():
        return "HYD"