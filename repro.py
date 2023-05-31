import collections.abc as collections_abc
import inspect

import wrapt

class TrackableDataStructure: pass

class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  """Wraps built-in dicts to support restore-on-create for variables.

  _DictWrapper is to Mapping as ListWrapper is to List. Unlike Mapping,
  _DictWrapper allows non-string keys and values and arbitrary mutations (delete
  keys, reassign values). Like ListWrapper, these mutations mean that
  _DictWrapper will raise an exception on save.
  """

  def __init__(self, wrapped_dict=None):
    if wrapped_dict is None:
      # Allow zero-argument construction, e.g. from session.run's re-wrapping.
      wrapped_dict = {}
    if not isinstance(wrapped_dict, collections_abc.Mapping):
      # Allow construction from a sequence, e.g. from nest.pack_sequence_as.
      wrapped_dict = dict(wrapped_dict)
    wrapt.ObjectProxy.__init__(self, wrapped_dict)

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

  @property
  def _dirty(self):
    """Check if there has already been a mutation which prevents saving."""
    return (self._self_external_modification
            or self._self_non_string_key)


inspect.getattr_static(_DictWrapper(), 'bar')
