"""Utilities to support docstring validation."""

# Authors: The scikit-learn developers
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
    fget : callable, default=None
        Getter function passed to :class:`property`.
    fset : callable, default=None
        Setter function passed to :class:`property`.
    fdel : callable, default=None
        Deleter function passed to :class:`property`.
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
        """Return a copy of the descriptor using *fget* as getter.

        Parameters
        ----------
        fget : callable
            Getter function used by the resulting descriptor.

        Returns
        -------
        DocstringProperty
            Copy of the descriptor using ``fget`` as getter.
        """
        prop = type(self)(fget, self.fset, self.fdel, doc=fget.__doc__)
        return prop

    def setter(self, fset):
        """Return a copy of the descriptor using *fset* as setter.

        Parameters
        ----------
        fset : callable
            Setter function used by the resulting descriptor.

        Returns
        -------
        DocstringProperty
            Copy of the descriptor using ``fset`` as setter.
        """
        prop = type(self)(self.fget, fset, self.fdel, doc=self.__doc__)
        return prop

    def deleter(self, fdel):
        """Return a copy of the descriptor using *fdel* as deleter.

        Parameters
        ----------
        fdel : callable
            Deleter function used by the resulting descriptor.

        Returns
        -------
        DocstringProperty
            Copy of the descriptor using ``fdel`` as deleter.
        """
        prop = type(self)(self.fget, self.fset, fdel, doc=self.__doc__)
        return prop
