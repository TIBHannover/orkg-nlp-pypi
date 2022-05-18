"""
Includes custom-defined exceptions
"""


class ORKGNLPValidationError(Exception):
    """
    Indicates an incorrect input attempt by the caller.
    """

    def __init__(self, message):
        super().__init__(message)
