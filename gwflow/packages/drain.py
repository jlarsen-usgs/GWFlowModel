import numpy as np

from .stress_package import StressPakBase


class Drain(StressPakBase):
    """
    Constant head boundary condition package

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    gh_df : pd.DataFrame
        pandas dataframe of boundary condition data
    package_name : str
        user provided package name, default is "ghb".
    """
    def __init__(self, parent, drn_df, package_name="drn"):
        super().__init__(parent, package_name)

        self._drn_input = drn_df[self.data_columns()]
        self._nodes = self._parent.lrc_to_node(list(zip(drn_df["k"], drn_df["i"], drn_df["j"])))
        self._elev = self._drn_input["elev"].values
        self._cond = self._drn_input["cond"].values

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
            hold[self.nodes] > self._elev,
            self._cond * self._elev,
            0
        )
        return -1 * Qn

    @property
    def hcof(self):
        """
        Returns the head coefficient term that's added to the A matrix cross terms
        for the package
        """
        hold = self._parent.hold
        hcof = np.where(
            hold[self.nodes] > self._elev,
            self._cond,
            0
        )
        return -1 * hcof

    @staticmethod
    def data_columns():
        """
        Returns a list of data columns that must be included in the package input
        dataframe
        """
        return ["k", "i", "j", "elev", "cond"]

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "GHB"
