""" Defines decorators used within the package """


def singleton(cls):
    """
    Reflects the singleton pattern and can be used to decorate classes.

    Only one instance can be obtained from a singleton class.

    :param cls: The decorated class.
    :return: Function responsible for returning a pre-initiated instance of the given ``cls``.
    """

    instances = {}

    def getinstance(**kwargs):
        if cls not in instances:
            instances[cls] = cls(**kwargs)
        return instances[cls]

    return getinstance
