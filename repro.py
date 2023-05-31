import collections.abc as collections_abc
import inspect

import wrapt

class Trackable:
  def _name_based_attribute_restore(self, checkpoint):
    self._self_name_based_restores.add(checkpoint)
    if self._self_update_uid < checkpoint.restore_uid:
      checkpoint.eager_restore(self)
      self._self_update_uid = checkpoint.restore_uid

  def _add_variable_with_custom_getter(self,
                                       name,
                                       shape=None,
                                       dtype=...,
                                       initializer=None,
                                       getter=None,
                                       overwrite=False,
                                       **kwargs_for_getter):
    self._maybe_initialize_trackable()
    with ops.init_scope():
      if context.executing_eagerly():
        checkpoint_initializer = self._preload_simple_restoration(name=name)
      else:
        checkpoint_initializer = None
      if (checkpoint_initializer is not None and
          not (isinstance(initializer, CheckpointInitialValueCallable) and
               (initializer.restore_uid > checkpoint_initializer.restore_uid))):
        initializer = checkpoint_initializer
    new_variable = getter(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        **kwargs_for_getter)

    if not overwrite or isinstance(new_variable, Trackable):
      return self._track_trackable(new_variable, name=name, overwrite=overwrite)
    else:
      return new_variable

  def _preload_simple_restoration(self, name):
    deferred_dependencies_list = self._deferred_dependencies.get(name, ())
    if not deferred_dependencies_list:
      # Nothing to do; we don't have a restore for this dependency queued up.
      return
    for checkpoint_position in deferred_dependencies_list:
      if not checkpoint_position.is_simple_variable():
        # If _any_ pending restoration is too complicated to fit in an
        # initializer (because it has dependencies, or because there are
        # multiple Tensors to restore), bail and let the general tracking code
        # handle it.
        return None
    checkpoint_position = max(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid)
    return CheckpointInitialValueCallable(
        checkpoint_position=checkpoint_position)

  def _track_trackable(self, trackable, name, overwrite=False):
    self._maybe_initialize_trackable()
    if not isinstance(trackable, Trackable):
      raise TypeError(
          "Trackable._track_trackable() can only be used to track objects of "
          f"type Trackable. Got type {type(trackable)}.")
    if not getattr(self, "_manual_tracking", True):
      return trackable
    new_reference = TrackableReference(name=name, ref=trackable)
    current_object = self._lookup_dependency(name)
    if (current_object is not None and current_object is not trackable):
      if not overwrite:
        raise ValueError(
            f"Called Trackable._track_trackable() with name='{name}', "
            "but a Trackable with this name is already declared as a "
            "dependency. Names must be unique (or overwrite=True).")
      # This is a weird thing to do, but we're not going to stop people from
      # using __setattr__.
      for index, (old_name, _) in enumerate(
          self._self_unconditional_checkpoint_dependencies):
        if name == old_name:
          self._self_unconditional_checkpoint_dependencies[
              index] = new_reference
    elif current_object is None:
      self._self_unconditional_checkpoint_dependencies.append(new_reference)
      self._handle_deferred_dependencies(name=name, trackable=trackable)
    self._self_unconditional_dependency_names[name] = trackable
    return trackable

  def _handle_deferred_dependencies(self, name, trackable):
    self._maybe_initialize_trackable()
    trackable._maybe_initialize_trackable()  # pylint: disable=protected-access
    deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    for checkpoint_position in sorted(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid,
        reverse=True):
      checkpoint_position.restore(trackable)

    for name_based_restore in sorted(
        self._self_name_based_restores,
        key=lambda checkpoint: checkpoint.restore_uid,
        reverse=True):
      trackable._name_based_attribute_restore(name_based_restore)  # pylint: disable=protected-access

  def _serialize_to_proto(self, object_proto=None, **kwargs):
    del object_proto, kwargs
    return None

  @classmethod
  def _deserialize_from_proto(cls,
                              proto=None,
                              dependencies=None,
                              object_proto=None,
                              export_dir=None,
                              asset_file_def=None,
                              operation_attributes=None,
                              **kwargs):
    del (proto, dependencies, object_proto, export_dir, asset_file_def,
         operation_attributes, kwargs)
    return cls()

  def _add_trackable_child(self, name, value):
    self._track_trackable(value, name, overwrite=True)

  def _deserialization_dependencies(self, children):
    del children  # Unused.
    return {}

  def _trackable_children(self,
                          save_type=...,
                          cache=None,
                          **kwargs):
    del save_type, cache, kwargs
    self._maybe_initialize_trackable()
    return {name: ref for name, ref in self._checkpoint_dependencies}

  def _export_to_saved_model_graph(self,
                                   object_map,
                                   tensor_map,
                                   options,
                                   **kwargs):
    _, _, _ = object_map, tensor_map, options
    del kwargs
    return []

class TrackableDataStructure(Trackable):
  def __init__(self):
    # Attributes prefixed with "_self_" for compatibility with
    # wrapt.ObjectProxy. All additional attrs MUST conform to this pattern, as
    # extending `__slots__` on a subclass of ObjectProxy breaks in a variety of
    # ways.
    self._self_trainable = True
    self._self_extra_variables = []

  @property
  def _attribute_sentinel(self):
    return self._self_attribute_sentinel

  @property
  def trainable(self):
    return self._self_trainable

  @trainable.setter
  def trainable(self, value):
    self._self_trainable = value

  def _track_value(self, value, name):
    value = sticky_attribute_assignment(
        trackable=self, value=value, name=name)
    if isinstance(value, variables.Variable):
      self._self_extra_variables.append(value)
    if not isinstance(value, base.Trackable):
      raise _UntrackableError(value)
    if hasattr(value, "_use_resource_variables"):
      value._use_resource_variables = True  # pylint: disable=protected-access
    value_attribute_sentinel = getattr(value, "_attribute_sentinel", None)
    if value_attribute_sentinel:
      value_attribute_sentinel.add_parent(self._attribute_sentinel)
    return value

  @property
  def _values(self):
    """An iterable/sequence which may contain trackable objects."""
    raise NotImplementedError("Abstract method")

  @property
  def _layers(self):
    """All Layers and Layer containers, including empty containers."""
    # Filter objects on demand so that wrapper objects use values from the thing
    # they're wrapping if out of sync.
    collected = []
    for obj in self._values:
      if (isinstance(obj, TrackableDataStructure)
          or layer_utils.is_layer(obj)
          or layer_utils.has_weights(obj)):
        collected.append(obj)
    return collected

  @property
  def layers(self):
    return list(layer_utils.filter_empty_layer_containers(self._layers))

  @property
  def trainable_weights(self):
    if not self._self_trainable:
      return []
    trainable_variables = []
    for obj in self._values:
      if isinstance(obj, base.Trackable) and hasattr(
          obj, "trainable_variables"):
        trainable_variables += obj.trainable_variables
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    return trainable_variables + trainable_extra_variables

  @property
  def non_trainable_weights(self):
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    non_trainable_extra_variables = [
        v for v in self._self_extra_variables if not v.trainable
    ]
    non_trainable_variables = []
    for obj in self._values:
      if isinstance(obj, base.Trackable) and hasattr(
          obj, "non_trainable_variables"):
        non_trainable_variables += obj.non_trainable_variables

    if not self._self_trainable:
      # Return order is all trainable vars, then all non-trainable vars.
      trainable_variables = []
      for obj in self._values:
        if isinstance(obj, base.Trackable) and hasattr(
            obj, "trainable_variables"):
          trainable_variables += obj.trainable_variables

      non_trainable_variables = (
          trainable_variables + trainable_extra_variables +
          non_trainable_variables + non_trainable_extra_variables)
    else:
      non_trainable_variables = (
          non_trainable_variables + non_trainable_extra_variables)

    return non_trainable_variables

  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def variables(self):
    return self.weights

  @property
  def updates(self):
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "updates"):
        aggregated += layer.updates
    return aggregated

  @property
  def losses(self):
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "losses"):
        aggregated += layer.losses
    return aggregated

  def __hash__(self):
    # Support object-identity hashing, so these structures can be used as keys
    # in sets/dicts.
    return id(self)

  def __eq__(self, other):
    # Similar to Tensors, trackable data structures use object-identity
    # equality to support set/dict membership.
    return self is other

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
    TrackableDataStructure.__init__(self)
    self._self_non_string_key = False
    self._self_external_modification = False
    self.__wrapped__.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self.__wrapped__.items()})
    self._update_snapshot()

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
  def _values(self):
    """Collect values for TrackableDataStructure."""
    # Sort items deterministically by key
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []

  def _trackable_children(self, save_type=..., **kwargs):
    """Check that the object is saveable before listing its dependencies."""
    self._check_self_external_modification()
    if self._self_non_string_key:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary "
          "contains a non-string key which maps to a trackable object or "
          "mutable data structure.\n\nIf you don't need this dictionary "
          "checkpointed, wrap it in a non-trackable "
          "object; it will be subsequently ignored.")
    if self._self_external_modification:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary was "
          f"modified outside the wrapper (its final value was {self}, its value"
          " when a checkpoint dependency was added was "
          f"{self._self_last_wrapped_dict_snapshot}), which breaks "
          "restoration on object creation.\n\nIf you don't need this "
          "dictionary checkpointed, wrap it in a "
          "non-trackable object; it will be subsequently ignored.")
    assert not self._dirty  # Any reason for dirtiness should have an exception.
    children = super()._trackable_children(save_type, **kwargs)

    if save_type == base.SaveType.SAVEDMODEL:
      # Add functions to be serialized.
      children.update(
          {key: value for key, value in self.items() if _is_function(value)})

    return children

  @property
  def _dirty(self):
    """Check if there has already been a mutation which prevents saving."""
    return (self._self_external_modification
            or self._self_non_string_key)

  def _check_self_external_modification(self):
    """Checks for any changes to the wrapped dict not through the wrapper."""
    if self._dirty:
      return
    if self != self._self_last_wrapped_dict_snapshot:
      self._self_external_modification = True
      self._self_last_wrapped_dict_snapshot = None

  def _update_snapshot(self):
    """Acknowledges tracked changes to the wrapped dict."""
    if self._dirty:
      return
    self._self_last_wrapped_dict_snapshot = dict(self)

  def _track_value(self, value, name):
    """Allows storage of non-trackable objects."""
    if isinstance(name, str):
      string_key = True
    else:
      name = "-non_string_key"
      string_key = False
    try:
      no_dependency = isinstance(value, NoDependency)
      value = super()._track_value(value=value, name=name)
      if not (string_key or no_dependency):
        # A non-string key maps to a trackable value. This data structure
        # is not saveable.
        self._self_non_string_key = True
      return value
    except ValueError:
      # Even if this value isn't trackable, we need to make sure
      # NoDependency objects get unwrapped.
      return sticky_attribute_assignment(
          trackable=self, value=value, name=name)

  def __setitem__(self, key, value):
    """Allow any modifications, but possibly mark the wrapper as unsaveable."""
    self._check_self_external_modification()
    self._maybe_initialize_trackable()
    no_dep = isinstance(value, NoDependency)
    if isinstance(key, str):
      value = self._track_value(value, name=key)
    else:
      value = wrap_or_unwrap(value)
      if not no_dep and isinstance(value, base.Trackable):
        # Non-string keys are OK as long as we have no reason to add a
        # dependency on the value (either because the value is not
        # trackable, or because it was wrapped in a NoDependency object).
        self._self_non_string_key = True
    self.__wrapped__[key] = value

    self._update_snapshot()

  def __delitem__(self, key):
    self._check_self_external_modification()
    del self.__wrapped__[key]
    self._update_snapshot()

  def __eq__(self, other):
    # Override the TrackableDataStructure "== -> is" forwarding and go back to
    # the wrapt implementation.
    return self.__wrapped__ == other

  def update(self, *args, **kwargs):
    for key, value in dict(*args, **kwargs).items():
      self[key] = value


inspect.getattr_static(_DictWrapper(), 'bar')
