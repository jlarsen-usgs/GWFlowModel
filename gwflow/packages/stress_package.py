import abc
from .package import PakBase


class StressPakBase(PakBase):
    """
    Base class for Stress Packages (boundary condition packages

    Parameters
    ----------
    parent : GroundwaterFlow
        Groundwater flow model instance
    package_name : str
        User supplied package name, most packages have a default
    """
    def __init__(self, parent, package_name):
        super().__init__(parent, package_name)


    @property
    @abc.abstractmethod
    def nodes(self):
        raise NotImplementedError("Must be defined in child class")


    @property
    @abc.abstractmethod
    def rhs(self):
        raise NotImplementedError("Must be defined in child class")

    @property
    @abc.abstractmethod
    def hcof(self):
        raise NotImplementedError("Must be defined in child class")

    @staticmethod
    @abc.abstractmethod
    def data_columns():
        raise NotImplementedError("Must be defined in child class")

    @staticmethod
    def package_type():
        raise NotImplementedError("Must be defined in child class")