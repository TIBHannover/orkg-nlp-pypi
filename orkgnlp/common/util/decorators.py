""" Defines decorators used within the package """


def singleton(_):
    """
    This decorator can only be used above the __new__ function of a class. It's responsible for returning a pre-created
    instance of the respective class or create a new one, if not have happened before.

    :param _: The __new__ function.
    :return:
    """

    def apply_pattern(cls, *args, **kwargs):
        # attention: *args and **kwargs must be included even if not used!
        if not hasattr(cls, 'instance'):
            cls.instance = super(cls.__class__, cls).__new__(cls)
        return cls.instance

    return apply_pattern
