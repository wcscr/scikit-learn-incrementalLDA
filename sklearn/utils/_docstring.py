"""Utilities to support docstring validation."""

# SPDX-License-Identifier: BSD-3-Clause


class DocstringProperty(property):
    """Property descriptor exposing metadata for docstring validators.

    Subclasses :class:`property` to preserve the regular descriptor behaviour
    while ensuring that the resulting object exposes ``__module__`` and
    ``__name__`` attributes. Some tooling, notably :mod:`numpydoc`, relies on
    this metadata to identify the fully qualified name of the documented
    object.

    Parameters
    ----------
    fget, fset, fdel : callable, default=None
        Accessor functions passed to :class:`property`.
    doc : str, default=None
        Docstring for the property. Falls back to ``fget.__doc__`` when
        omitted.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        self._set_metadata(fget, fset, fdel)

    def _set_metadata(self, fget, fset, fdel):
        func = fget or fset or fdel
        if func is not None:
            self.__module__ = func.__module__
            self.__name__ = func.__name__
            if hasattr(func, "__qualname__"):
                self.__qualname__ = func.__qualname__

    def getter(self, fget):
        prop = type(self)(fget, self.fset, self.fdel, doc=fget.__doc__)
        return prop

    def setter(self, fset):
        prop = type(self)(self.fget, fset, self.fdel, doc=self.__doc__)
        return prop

    def deleter(self, fdel):
        prop = type(self)(self.fget, self.fset, fdel, doc=self.__doc__)
        return prop
