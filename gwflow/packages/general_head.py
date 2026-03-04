import numpy as np
from .stress_package import StressPakBase


class GeneralHead(StressPakBase):
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
    def __init__(self, parent, gh_df, package_name="ghb"):
        super().__init__(parent, package_name)

        self._ghb_input = gh_df[self.data_columns()]
        self._nodes = self._parent.lrc_to_node(list(zip(gh_df["k"], gh_df["i"], gh_df["j"])))
        self._heads = self._ghb_input["elev"].values
        self._cond = self._ghb_input["cond"].values

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
        Qn = self._cond * self._heads
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
        return ["k", "i", "j", "elev", "cond"]

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "GHB"
