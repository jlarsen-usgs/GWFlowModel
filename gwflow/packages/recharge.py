import numpy as np
from .stress_package import StressPakBase


class Recharge(StressPakBase):
    """
    Recharge boundary condition package

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    rch_array : numpy array
        numpy array of boundary condition data
    rch_layer : np.array or None
        if None recharge is applied to the top layer (layer 0)
    package_name : str
        user provided package name, default is "rch".
    """
    def __init__(self, parent, rch_array, rch_layer=None, package_name="rch"):
        super().__init__(parent, package_name)

        rch_array = rch_array.reshape((self._parent.nrow, self._parent.ncol))

        self._rch_array = rch_array
        if rch_layer is None:
            rch_layer = np.zeros((self._parent.nrow, self._parent.ncol), dtype=int)
        self._rch_layer = rch_layer

        lrcs = []
        for i in range(self._parent.nrow):
            for j in range(self._parent.ncol):
                lrcs.append((rch_layer[i, j], i, j))

        self._nodes = self._parent.lrc_to_node(lrcs)

        self._rch_array = self._rch_array.ravel()
        self._rch_layer = self._rch_layer.ravel()
        self._cell_area = self._parent._dis.cell_area

    @property
    def nodes(self):
        """
        Returns a numpy array of the boundary condition node numbers
        """
        return self._nodes

    @property
    def rhs(self):
        """
        Returns the right hand side term for the package for the CVFD solution
        """
        Qn = self._rch_array * self._cell_area[self._nodes]
        return -1 * Qn

    @property
    def hcof(self):
        """
        Returns the head coefficient term that's added to the A matrix cross terms
        for the package
        """
        return np.zeros((len(self._nodes)), dtype=float)

    @staticmethod
    def data_columns():
        """
        Returns a list of data columns that must be included in the package input
        dataframe
        """
        return None

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "RCH"