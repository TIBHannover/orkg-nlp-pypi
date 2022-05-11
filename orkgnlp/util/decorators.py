""" Defines decorators used within the package """


def singleton(cls):
    """
    Reflects the singleton pattern and can be used to decorate classes.

    Only one instance can be obtained from a singleton class.

    :param cls:
    :return:
    """

    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance
