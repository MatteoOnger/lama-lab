import copy


def deep_update(base_dict: dict, override_dict: dict) -> dict:
    """
    Performs a deep merge of two dictionaries.
    Values from override_dict overwrite those in base_dict without destroying nested structures.
    """
    updated = copy.deepcopy(base_dict)
    for k, v in override_dict.items():
        if k in updated and isinstance(updated[k], dict) and isinstance(v, dict):
            updated[k] = deep_update(updated[k], v)
        else:
            updated[k] = copy.deepcopy(v)
    return updated
