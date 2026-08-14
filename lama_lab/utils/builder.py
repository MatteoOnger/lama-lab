import importlib
import copy


def build_from_config(config):
    """
    Recursively instantiates objects from a configuration dictionary.
    If a dictionary contains a '_target_' key, it imports the specified
    callable and initializes it with the remaining key-value pairs.
    """
    if not isinstance(config, dict):
        return config

    cfg = copy.deepcopy(config)

    if "_target_" not in cfg:
        return {k: build_from_config(v) for k, v in cfg.items()}

    target_path: str = cfg.pop("_target_")
    module_path, obj_name = target_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    target_callable = getattr(module, obj_name)

    kwargs = {k: build_from_config(v) for k, v in cfg.items()}

    # kwargs -> instantiate/call;
    # no kwargs -> pass the function/class itself
    if kwargs:
        return target_callable(**kwargs)
    else:
        return target_callable
