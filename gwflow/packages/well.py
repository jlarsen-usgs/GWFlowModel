import numpy as np

from .stress_package import StressPakBase


class Well(StressPakBase):
    """
    Constant head boundary condition package

    Parameters
    ----------
    parent : GroundwaterFlow
        groundwater flow model instance
    gh_df : pd.DataFrame
        pandas dataframe of boundary condition data
    package_name : str
        user provided package name, default is "well".
    """
    def __init__(self, parent, wel_df, package_name="well"):
        super().__init__(parent, package_name)

        self._ghb_input = wel_df[self.data_columns()]
        self._nodes = self._parent.lrc_to_node(list(zip(wel_df["k"], wel_df["i"], wel_df["j"])))
        self._flux = self._ghb_input["flux"].values

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
        return -1 * self._flux

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
        return ["k", "i", "j", "q"]

    @staticmethod
    def package_type():
        """
        Returns the specific package type acronym
        """
        return "WEL"
