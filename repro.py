import collections.abc as collections_abc
import inspect

import wrapt

class TrackableDataStructure: pass

class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  def __init__(self):
    wrapt.ObjectProxy.__init__(self, {})


inspect.getattr_static(_DictWrapper(), 'bar')
