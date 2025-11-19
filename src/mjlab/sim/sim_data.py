"""Wraps a mujoco warp simulation to expose the arrays as torch tensors."""

import enum
from typing import Any, Generic, Optional, Sequence, TypeVar, Union

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


T = TypeVar("T")


def _add_torch_operators(cls):
  """Adds pytorch operators to a class."""
  two_sided_ops = ("add", "sub", "mul", "div", "pow")
  one_sided_ops = ("eq", "ne", "lt", "le", "gt", "ge")

  for op in one_sided_ops + two_sided_ops:
    torch_fn = getattr(torch, op)
    setattr(cls, f"__{op}__", lambda s, o, func=torch_fn: func(s.tensor, o))
  for op in two_sided_ops:
    torch_fn = getattr(torch, op)
    setattr(cls, f"__r{op}__", lambda s, o, func=torch_fn: func(o, s.tensor))

  setattr(cls, "__pos__", lambda s: +s.tensor)
  setattr(cls, "__neg__", lambda s: -s.tensor)
  setattr(cls, "__abs__", lambda s: torch.abs(s.tensor))
  setattr(cls, "__truediv__", lambda s, o: s.tensor / o)
  setattr(cls, "__rtruediv__", lambda s, o: o / s.tensor)
  return cls


class NoSetAttr:

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError(
        "This wrapper is read-only so that Warp array pointers are kept"
        " immutable for usage in CUDA graphs. Use in-place torch operations"
        " instead."
    )


def _is_batched_wp_array(wp_array: wp.array) -> bool:
  """Checks if a Warp array is a batched mjw.Model field."""
  return hasattr(wp_array, "_is_batched") and wp_array._is_batched  # pylint: disable=protected-access


@_add_torch_operators
class TorchTensor(NoSetAttr):
  """Wraps a warp/numpy array to expose it as a torch.Tensor."""

  def __init__(self, wp_array: wp.array):
    if isinstance(wp_array, np.ndarray):
      with wp.ScopedDevice("cpu"):
        wp_array = wp.from_numpy(wp_array)

    object.__setattr__(self, "_wp_array", wp_array)  # store the original array
    object.__setattr__(self, "_torch_tensor", wp.to_torch(wp_array))
    object.__setattr__(self, "_torch_stream", self._get_torch_stream())

  def _get_torch_stream(self) -> Optional[torch.cuda.Stream]:
    """Creates a torch stream from warp device stream."""
    if self._wp_array.device.is_cpu:
      return None
    stream = wp.get_stream(self._wp_array.device)
    return torch.cuda.ExternalStream(stream.cuda_stream)

  @property
  def wp_array(self) -> wp.array:
    return self._wp_array

  @property
  def tensor(self) -> torch.Tensor:
    return self._torch_tensor

  @property
  def device(self) -> str:
    return self._torch_tensor.device

  def __repr__(self) -> str:
    return repr(self._torch_tensor)

  def __getitem__(self, idx: Any) -> Any:
    return self._torch_tensor[idx]

  def __setitem__(self, idx: Any, value: Any):
    if self._wp_array.device.is_cpu:
      self._torch_tensor[idx] = value
      return

    if _is_batched_wp_array(self._wp_array) and self._wp_array.strides[0] == 0:
      raise AssertionError("Underlying Warp array is batched and has stride 0.")

    with torch.cuda.stream(self._torch_stream):
      self._torch_tensor[idx] = value

  def __getattr__(self, name: str) -> Any:
    return getattr(self._torch_tensor, name)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    """Allows torch functions to be called on TorchTensor."""
    if kwargs is None:
      kwargs = {}

    if not any(issubclass(t, cls) for t in types):
      return NotImplementedError(
          f"TorchTensor.__torch_function__ called with types {types}"
      )

    def _get_tensor(x: Any) -> Any:
      if isinstance(x, cls):
        return x.tensor
      elif isinstance(x, (list, tuple)):
        return type(x)(_get_tensor(y) for y in x)
      else:
        return x

    args = tuple(_get_tensor(arg) for arg in args)
    kwargs = {k: _get_tensor(v) for k, v in kwargs.items()}
    return func(*args, **kwargs)


def _tile_field(wp_array: wp.array, nworld: int) -> wp.array:
  """Tiles fields that are stride 0.

  Args:
    wp_array: warp array to tile.
    nworld: the number of worlds to tile the field to.

  Returns:
    The tiled warp array.
  """
  src = wp.to_torch(wp_array)
  dst = src.repeat((nworld,) + (1,) * (src.ndim - 1))
  dst_wp_array = wp.from_torch(dst)
  dst_wp_array._is_batched = True  # pylint: disable=protected-access
  return dst_wp_array


def _obj_has_array(obj: Any) -> bool:
  """Checks recursively if an object contains Warp arrays."""
  if isinstance(obj, wp.array) or isinstance(obj, np.ndarray):
    return True

  for attr in getattr(obj, "__dict__", {}).keys():
    if _obj_has_array(getattr(obj, attr)):
      return True

  return False


class TorchWrapper(Generic[T], NoSetAttr):
  """Wraps objects that contain Warp/Numpy arrays and exposes them as tensors."""

  def __init__(self, obj: T, nworld: int | None = None) -> None:
    if isinstance(obj, mjw.Data):
      nworld = obj.nworld
    if isinstance(obj, mjw.Model) and nworld is None:
      raise ValueError("nworld must be provided when wrapping a Model object.")

    object.__setattr__(self, "_obj", obj)
    object.__setattr__(self, "_tensor_cache", {})
    object.__setattr__(self, "_nworld", nworld or 1)
    object.__setattr__(self, "_bound_cache", {})

  def __getattr__(self, name: str) -> Any:
    if name in self._tensor_cache:
      return self._tensor_cache[name]

    val = getattr(self._obj, name)

    if (
        isinstance(val, wp.array)
        and _is_batched_wp_array(val)
        and val.strides[0] == 0
        and isinstance(self._obj, mjw.Model)
    ):
      # tile model fields with stride 0. we may consider leaving the stride 0
      # and tiling as needed in the future.
      val = _tile_field(val, self._nworld)
      setattr(self._obj, name, val)

    if isinstance(val, wp.array) or isinstance(val, np.ndarray):
      torch_val = TorchTensor(val)
      self._tensor_cache[name] = torch_val
      return torch_val

    if _obj_has_array(val):
      wrapped_val = TorchWrapper(val, self._nworld)
      self._tensor_cache[name] = wrapped_val
      return wrapped_val

    self._tensor_cache[name] = val
    return val

  def __repr__(self) -> str:
    return f"TorchWrapper({repr(self._obj)})"

  @property
  def obj(self) -> T:
    return self._obj

  @property
  def nworld(self) -> int:
    return self._nworld

  def bind(self, *args) -> Union["BoundDataAccessor", "BoundModelAccessor"]:
    """Binds a Data or Model object."""
    if isinstance(self._obj, mjw.Data):
      # Usage: data.bind(model, specs)
      model, specs = args[0], args[1]
      specs = (specs,) if not isinstance(specs, (list, tuple)) else tuple(specs)

      # TODO(btaba): is this cache key sufficient for all possible specs?
      # TODO(btaba): add model signature to the key
      key = ("data", type(specs[0]), tuple(s.id for s in specs))
      if key in self._bound_cache:
        return self._bound_cache[key]

      if not isinstance(model, TorchWrapper) or not isinstance(
          model.obj, mjw.Model
      ):
        raise TypeError(
            "bind on a Data object requires a model wrapper as the first"
            f" argument, got {model}."
        )
      self._bound_cache[key] = BoundDataAccessor(self, model, specs)
      return self._bound_cache[key]

    if isinstance(self._obj, mjw.Model):
      # Usage: model.bind(spec)
      specs = args[0]
      specs = (specs,) if not isinstance(specs, (list, tuple)) else tuple(specs)

      # TODO(btaba): is this cache key sufficient for all possible specs?
      # TODO(btaba): add model signature to the key
      key = ("model", type(specs[0]), tuple(s.id for s in specs))
      if key in self._bound_cache:
        return self._bound_cache[key]

      self._bound_cache[key] = BoundModelAccessor(self, specs)
      return self._bound_cache[key]

    raise TypeError(f"bind() is not supported for wrapped object '{self._obj}'")


class JointType(enum.IntEnum):
  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class BoundDataAccessor:
  """Binds a Data object."""

  def __init__(
      self,
      data_wrapper: "TorchWrapper",
      model_wrapper: "TorchWrapper",
      specs: Union[Sequence[mujoco.MjStruct], mujoco.MjStruct],
  ):
    self.data = data_wrapper
    self.model = model_wrapper
    self.prefix = ""

    ids = []
    for spec in specs:
      # TODO(btaba): check model signature against spec signature
      if spec.id < 0:
        raise ValueError(f"Invalid spec ID: {spec.id}.")

      if spec.id < 0:
        raise KeyError(f"invalid id: {spec.id}")
      elif isinstance(spec, mujoco.MjsBody):
        pass
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = "jnt_"
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = "geom_"
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = "site_"
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = "light_"
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = "cam_"
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = "ten_"
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = "actuator_"
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = "sensor_"
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = "eq_"
      else:
        raise ValueError("invalid spec type")
      ids.append(spec.id)

    if len(ids) == 1:
      ids = ids[0]

    self.device = data_wrapper.qpos.device
    self.id = torch.tensor(ids, dtype=torch.int, device=self.device)
    self._index_cache = {}

  def __getname(self, name: str):  # pylint: disable=unused-private-member
    """Get the name of the attribute and check if the type is correct."""
    if name == "sensordata":
      if self.prefix == "sensor_":
        return name
      raise AttributeError("sensordata is not available for this type")
    if name == "ctrl":
      if self.prefix == "actuator_":
        return name
      raise AttributeError("ctrl is not available for this type")
    if (
        name == "qpos"
        or name == "qvel"
        or name == "qacc"
        or name.startswith("qfrc_")
    ):
      if self.prefix == "jnt_":
        return name
      raise AttributeError(
          "qpos, qvel, qacc, qfrc are not available for this type"
      )
    return self.prefix + name

  def _get_indices(self, name: str) -> torch.Tensor:
    """Get the indices for the bound object(s) across all worlds."""
    if name in ("sensordata", "qpos", "qvel", "qacc") or (
        name.startswith("qfrc_")
    ):
      adr = num = 0
      if name == "sensordata":
        adr = self.model.sensor_adr[self.id]
        num = self.model.sensor_dim[self.id]
      elif name == "qpos":
        adr = self.model.jnt_qposadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.qpos_width() for jt in JointType)
      elif name == "qvel" or name == "qacc" or name.startswith("qfrc_"):
        adr = self.model.jnt_dofadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.dof_width() for jt in JointType)

      if self.id.shape:
        idx = []
        for a, n in zip(adr, num):
          idx.extend(a + j for j in range(n))
        return torch.tensor(idx, dtype=torch.int, device=self.device)

      if num > 1:
        return torch.arange(adr, adr + num, dtype=torch.int, device=self.device)
      return adr

    if name in ("mocap_pos", "mocap_quat"):
      return self.model.body_mocapid[self.id]

    return self.id

  def _get_indices_cached(self, name: str) -> torch.Tensor:
    if name in self._index_cache:
      return self._index_cache[name]
    idx = self._index_cache[name] = self._get_indices(name)
    return idx

  def __getattr__(self, name: str) -> torch.Tensor:
    """Gets data for the bound object(s) across all worlds."""
    target_tensor = getattr(self.data, self.__getname(name))
    indices = self._get_indices_cached(name)
    # NB: the returned tensor will not mutate the underlying data.
    return target_tensor[:, indices]

  def set(
      self,
      name: str,
      value: torch.Tensor,
      env_ids: Union[torch.Tensor, slice, None] = None,
  ) -> None:
    """Sets data for the bound object(s) across all worlds."""
    target_tensor = getattr(self.data, self.__getname(name))
    indices = self._get_indices_cached(name)
    if env_ids is None:
      env_ids = slice(None)

    is_indices_tensor = isinstance(indices, torch.Tensor) and indices.ndim > 0
    is_env_ids_tensor = isinstance(env_ids, torch.Tensor) and env_ids.ndim > 0

    if is_indices_tensor and is_env_ids_tensor:
      env_grid, id_grid = torch.meshgrid(env_ids, indices, indexing="ij")
      target_tensor[env_grid, id_grid] = value
    else:
      target_tensor[env_ids, indices] = value


class BoundModelAccessor:
  """Binds a Model object."""

  def __init__(
      self,
      model_wrapper: "TorchWrapper",
      specs: Union[Sequence[mujoco.MjStruct], mujoco.MjStruct],
  ):
    self.model = model_wrapper
    self.prefix = ""

    ids = []
    for spec in specs:
      # TODO(btaba): check model signature against spec signature
      if spec.id < 0:
        raise ValueError(f"Invalid spec ID: {spec.id}.")

      if spec.id < 0:
        raise KeyError(f"invalid id: {spec.id}")
      elif isinstance(spec, mujoco.MjsBody):
        self.prefix = "body_"
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = "jnt_"
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = "geom_"
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = "site_"
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = "light_"
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = "cam_"
      elif isinstance(spec, mujoco.MjsMesh):
        self.prefix = "mesh_"
      elif isinstance(spec, mujoco.MjsHField):
        self.prefix = "hfield_"
      elif isinstance(spec, mujoco.MjsPair):
        self.prefix = "pair_"
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = "tendon_"
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = "actuator_"
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = "sensor_"
      elif isinstance(spec, mujoco.MjsNumeric):
        self.prefix = "numeric_"
      elif isinstance(spec, mujoco.MjsText):
        self.prefix = "text_"
      elif isinstance(spec, mujoco.MjsTuple):
        self.prefix = "tuple_"
      elif isinstance(spec, mujoco.MjsKey):
        self.prefix = "key_"
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = "eq_"
      elif isinstance(spec, mujoco.MjsExclude):
        self.prefix = "exclude_"
      elif isinstance(spec, mujoco.MjsSkin):
        self.prefix = "skin_"
      elif isinstance(spec, mujoco.MjsMaterial):
        self.prefix = "material_"
      else:
        raise ValueError("invalid spec type")
      ids.append(spec.id)

    if len(ids) == 1:
      ids = ids[0]

    self.device = model_wrapper.geom_pos.device
    self.id = torch.tensor(ids, dtype=torch.int, device=self.device)

  def __getname(self, name: str) -> str:  # pylint: disable=unused-private-member
    return self.prefix + name

  def __getattr__(self, name: str) -> torch.Tensor:
    """Gets model field for the bound object(s) across all worlds."""
    name = self.__getname(name)
    target_tensor = getattr(self.model, name)
    # NB: the returned tensor will not mutate the underlying data.
    if _is_batched_wp_array(getattr(self.model.obj, name)):
      return target_tensor[:, self.id]
    return target_tensor[self.id]

  def set(
      self,
      name: str,
      value: torch.Tensor,
      env_ids: Union[torch.Tensor, slice, None] = None,
  ) -> None:
    """Sets model field for the bound object(s) across all worlds."""
    name = self.__getname(name)
    target_tensor = getattr(self.model, name)
    if not _is_batched_wp_array(getattr(self.model.obj, name)):
      target_tensor[self.id] = value
      return

    if env_ids is None:
      env_ids = slice(None)

    is_id_tensor = isinstance(self.id, torch.Tensor) and self.id.ndim > 0
    is_env_ids_tensor = isinstance(env_ids, torch.Tensor) and env_ids.ndim > 0

    if is_id_tensor and is_env_ids_tensor:
      env_grid, id_grid = torch.meshgrid(env_ids, self.id, indexing="ij")
      target_tensor[env_grid, id_grid] = value
    else:
      target_tensor[env_ids, self.id] = value
