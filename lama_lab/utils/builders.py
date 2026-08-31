import copy
import functools
import importlib
from typing import Any


def build_from_config(config: Any) -> Any:
    """Recursively instantiates objects from a configuration container.

    If a dictionary contains a '_target_' key, it dynamically imports the
    specified callable (class or function) and instantiates/invokes it with
    the remaining key-value pairs as keyword arguments.

    Supports deferred instantiation via '_partial_':
    - True or "partial": Returns a `functools.partial` object.
    - "dict" or "raw": Returns the raw configuration dictionary.

    Parameters
    ----------
    config : Any
        Configuration structure (dict, list, tuple, or primitive value).

    Returns
    -------
    Any
        Instantiated object, partial function, dictionary, list, or primitive value.

    Raises
    ------
    ValueError
        If '_target_' is not a valid dot-separated string path.

    Examples
    --------
    >>> cfg = {"_target_": "torch.zeros", "size": [2, 3]}
    >>> build_from_config(cfg)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    if isinstance(config, list):
        return [build_from_config(item) for item in config]
    if isinstance(config, tuple):
        return tuple(build_from_config(item) for item in config)
    if not isinstance(config, dict):
        return config

    cfg = copy.deepcopy(config)

    # Extract the deferred instantiation directive
    partial_mode = cfg.pop("_partial_", False)

    # Short-circuit: return raw dictionary for manual resolution
    if partial_mode in ("dict", "raw"):
        return cfg

    # Container dictionary without a target: recurse on nested values
    if "_target_" not in cfg:
        return {k: build_from_config(v) for k, v in cfg.items()}

    # Resolve target callable path
    target_path: str = cfg.pop("_target_")
    if not isinstance(target_path, str) or "." not in target_path:
        raise ValueError(
            f"'_target_' must be a dot-separated string path (e.g., 'module.Callable'). "
            f"Got: {target_path!r}"
        )

    module_path, obj_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target_callable = getattr(module, obj_name)

    # Recursively build keyword arguments
    kwargs = {k: build_from_config(v) for k, v in cfg.items()}

    # Deferred partial instantiation via functools.partial
    if partial_mode in (True, "partial"):
        return functools.partial(target_callable, **kwargs)

    # Immediate instantiation
    return target_callable(**kwargs)
