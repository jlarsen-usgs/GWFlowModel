import numpy as np
from .stress_package import StressPakBase


class ConstantHead(StressPakBase):
    """
    Constant head boundary condition package

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    chd_df : pd.DataFrame
        pandas dataframe of boundary condition data
    package_name : str
        user provided package name, default is "chd".

    """
    def __init__(self, parent, chd_df, package_name="chd"):
        super().__init__(parent, package_name)

        self._chd_input = chd_df[self.data_columns()]
        self._nodes = self._parent.lrc_to_node(list(zip(chd_df["k"], chd_df["i"], chd_df["j"])))
        self._heads = self._chd_input["head"].values

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
        return self._heads

    @property
    def hcof(self):
        """
        Returns the head coefficient term that's added to the A matrix cross terms
        for the package
        """
        return np.zeros(self._nodes.shape, dtype=float)

    @staticmethod
    def data_columns():
        """
        Returns a list of data columns that must be included in the package input
        dataframe
        """
        return ["k", "i", "j", "head"]

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "CHD"