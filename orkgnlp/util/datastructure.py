"""
Includes help classes useful for encapsulating data.
"""


class StrictDict(dict):
    """
    This class prevents adding new keys to the dictionary
    """

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError("{} is not a legal key of this StrictDict".format(repr(key)))
        dict.__setitem__(self, key, value)

    def update(self, __m, **kwargs):
        for key, value in __m.items():
            self.__setitem__(key, value)
