""" Defines decorators used within the package """
from typing import Type


def singleton(_):
    """
    This decorator can only be used above the __new__ function of a class. It's responsible for returning a pre-created
    instance of the respective class or create a new one, if not have happened before.

    :param _: The __new__ function.
    """

    def apply_pattern(cls: Type, *args, **kwargs):
        # attention: *args and **kwargs must be included even if not used!
        if not hasattr(cls, 'instance'):
            cls.instance = super(cls.__class__, cls).__new__(cls)
        return cls.instance

    return apply_pattern


def sanitize_text(f):
    """
    This decorator can only be used above functions having the argument ``text`` at first position. The wrapper function
    returns the passed text if it's not empty or an empty string in case of empty or None passed argument.

    :param f: The decorated function.
    """
    def wrapper(text: str, *args):
        if not text:
            return f('', *args)

        return f(text, *args)

    return wrapper
