import abc


class PakBase:
    """
    Base Class for all Packages

    Parameters
    ----------
    parent : GroundwaterFlow
        Groundwater flow model instance
    package_name : str
        User supplied package name, most packages have a default

    """

    def __init__(self, parent, package_name):
        self._parent = parent
        self._package_name = package_name
        self._parent.add_package(self)

    def parent_model(self):
        """
        Returns the parent model
        """
        return self._parent

    def package_name(self):
        """
        Returns the user supplied package name
        """
        return self._package_name.upper()

    @staticmethod
    @abc.abstractmethod
    def package_type():
        raise NotImplementedError("Must be defined in child class")




