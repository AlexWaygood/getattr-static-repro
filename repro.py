import collections.abc as collections_abc
import inspect

import wrapt

class TrackableDataStructure: pass

class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy): ...


inspect.getattr_static(_DictWrapper({}), 'bar')
