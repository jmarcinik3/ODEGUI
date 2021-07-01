"""
This class contains custom errors that are used frequently throughout the project.
"""
from typing import Any, List, Union


class RecursiveTypeError(TypeError):
    """
    This class treats a TypeError raised during recursion or recursion-esque methods.
    """

    def __init__(self, err_var: Any, safe_cl: Union[type, List[type]] = str, err_name: str = "input") -> None:
        """
        Constructor for :class:`~CustomErrors.RecursiveTypeError`
        
        :param err_var: variable that caused error to be raised
        :param safe_cl: accepted classes that won't cause error to be raised
        :param err_name: original name of error-causing variable
        """
        self.variable = err_var
        self.name = err_name
        self.safe_classes = [safe_cl] if not isinstance(safe_cl, list) else safe_cl
        super().__init__(self.getMessage())

    def getVariable(self) -> Any:
        """
        Get variable that caused error to be raised.
        """
        return self.variable

    def getVariableClass(self) -> str:
        """
        Get class of error-causing variable.
        """
        return self.getVariable().__class__.__name__

    def getVariableName(self) -> str:
        """
        Get original name of error-causing variable.
        """
        return self.name

    def getSafeClasses(self) -> str:
        """
        Get classes that prevent error from being raised.
        """
        cl_strs = [cl.__name__ for cl in self.safe_classes]
        cl_str = ', '.join(cl_strs)
        return cl_str

    def getMessage(self) -> str:
        """
        Get message with including (1) info about error-causing variable and (2) error-preventing classes.
        """
        var = self.getVariable()
        cl = self.getVariableClass()
        name = self.getVariableName()
        safe_cls = self.getSafeClasses()
        message = f"{name:s} ({cl:s}, {var:}) must be of type {safe_cls:s} or list thereof"
        return message
