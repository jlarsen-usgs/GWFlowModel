import numpy as np
from .stress_package import StressPakBase


class Evapotranspiration(StressPakBase):
    """
    Evapotranspiration boundary condition package

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    evt_array : numpy array
        numpy array of boundary condition data
    evt_surface : np.array
        elevation of the Evapotranspiration surface
    ext_depth : np.array
        extinction depth of the EVT surface
    package_name : str
        user provided package name, default is "evt".
    """
    def __init__(self, parent, evt_array, evt_surface, ext_depth, package_name="evt"):
        super().__init__(parent, package_name)

        self._evt_array = np.abs(evt_array.reshape((self._parent.ncpl,)))

        self._evt_surface = evt_surface.reshape((self._parent.ncpl,))
        self._ext_depth = np.abs(ext_depth.reshape((self._parent.ncpl,)))
        self._ext_surface = self._evt_surface - self._ext_depth

        self._nodes = np.arange(self._parent.ncpl, dtype=int)
        self._cell_area = self._parent._dis.cell_area
        self._evtr = self._evt_array * self._cell_area[self._nodes]

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
        # equation 6-30a,b,c Modflow 6 documentation

        hold = self._parent.hold

        # 6-30a if True, fill with 6-30c if False
        Qetnb = np.where(
            hold[self._nodes] > self._evt_surface,
            self._evtr,
            0
        )
        # 6-30b if True, else use previous calc from 6-30a or 6-30c
        # refactor equation 6-30b to
        #    Evtr * h       Evtr * surf     Evtr * Extdp
        #    --------   -   -----------  -  ------------
        #      Extdp           Extdp            Extdp
        #
        #    HCOF term  |          RHS TERMS
        Qetnb = np.where(
            (self._ext_surface <= hold[self._nodes]) & (hold[self._nodes] <= self._evt_surface),
            self._evtr - ((self._evtr * self._evt_surface) / self._ext_depth),
            Qetnb
        )

        Qetnb = np.where(
            hold[self._nodes] < self._ext_surface,
            0,
            Qetnb
        )

        return -1 * Qetnb

    @property
    def hcof(self):
        """
        Returns the head coefficient term that's added to the A matrix cross terms
        for the package
        """
        # HCOF calculation definition is in the comments of RHS
        hold = self._parent.hold

        hcof = np.where(
            hold[self._nodes] > self._ext_surface,
            self._evtr / self._ext_depth,
            0
        )

        return -1 * hcof

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
        return "EVT"