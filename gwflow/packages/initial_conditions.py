import numpy as np
from .package import PakBase


# might want to make this a "Head package"
class InitialConditions(PakBase):
    """
    Package to define the Hydraulic parameters of a GroundwaterFlow model

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    strt : np.ndarray
        array of starting head values for the model
    package_name : str
        user provided package name, default is "ic".
    """

    def __init__(self, parent, strt, package_name="hyd"):
        super().__init__(parent, package_name)
        self._strt = np.ravel(strt)

    @property
    def strt(self):
        """

        Returns the starting heads

        """
        return self._strt

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "IC"
