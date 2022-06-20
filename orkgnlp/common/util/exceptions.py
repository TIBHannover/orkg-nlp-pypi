"""
Includes custom-defined exceptions
"""


class ORKGNLPValidationError(Exception):
    """
    Indicates an incorrect input attempt by the caller.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ORKGNLPIllegalStateError(RuntimeError):
    """
    Indicates an incorrect order of function calls by the caller.
    """

    def __init__(self, message: str):
        super().__init__(message)
