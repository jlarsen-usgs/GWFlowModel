import numpy as np
from .stress_package import StressPakBase


class River(StressPakBase):
    """
    River boundary condition pacakge

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    riv_df : pd.DataFrame
        pandas dataframe of boundary condition data
    package_name : str
        user provided package name, default is "riv".
    """
    def __init__(self, parent, riv_df, package_name="riv"):
        super().__init__(parent, package_name)
        self._riv_input = riv_df[self.data_columns()]
        self._nodes = self._parent.lrc_to_node(list(zip(riv_df["k"], riv_df["i"], riv_df["j"])))
        self._heads = self._riv_input["stage"].values
        self._cond = self._riv_input["cond"].values
        self._rbot = self._riv_input["rbot"].values

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
        hold = self._parent.hold
        Qn = np.where(
            hold[self.nodes] > self._rbot,
            self._cond * self._heads, # MF6 eq. 6-25a
            self._cond * (self._heads - self._rbot) # MF6 eq. 6-25b
        )
        return -1 * Qn

    @property
    def hcof(self):
        """
        Returns the head coefficient term that's added to the A matrix cross terms
        for the package
        """
        return -1 * self._cond

    @staticmethod
    def data_columns():
        """
        Returns a list of data columns that must be included in the package input
        dataframe
        """
        return ["k", "i", "j", "stage", "cond", "rbot"]

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "RIV"
