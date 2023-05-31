import collections.abc as collections_abc
import inspect

import wrapt

class TrackableDataStructure: pass

class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  def __init__(self):
    wrapt.ObjectProxy.__init__(self, {})

  def __getattribute__(self, name):
    if (hasattr(type(self), name)
        and isinstance(getattr(type(self), name), property)):
      # Bypass ObjectProxy for properties. Whether this workaround is necessary
      # appears to depend on the Python version but not the wrapt version: 3.4
      # in particular seems to look up properties on the wrapped object instead
      # of the wrapper without this logic.
      return object.__getattribute__(self, name)
    else:
      return super().__getattribute__(name)


inspect.getattr_static(_DictWrapper(), 'bar')
