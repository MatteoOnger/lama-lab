---
name: python-docstrings-comments-polish
description: Polish docstrings and inline comments in existing Python code (NumPy-style docstrings, plain-text code references, purposeful comments) without changing any logic. Use when the user asks to improve, clean up, document, or comment existing Python code, or to review/fix docstrings, without altering behavior.
---

# Python Docstrings and Comments Polish

Polish existing Python code by improving **docstrings and inline comments** for clarity, accuracy, and completeness **without modifying any executable code or logic**.

## Critical constraint: preserve code and logic

This is the highest-priority rule.

- **Do NOT touch any Python code:** Do not change variable names, control flow, functions, method signatures, return statements, type hints, exception types, or logic.
- Only add, modify, or remove **docstrings** and **comments**.
- When in doubt, preserve the existing code exactly as it is.

---

## Docstrings

Use concise, precise **NumPy-style docstrings**.

### General rules

- Public classes, functions, and methods should have appropriate docstrings.
- When you find a sequence of """""""" with no content, add a meaningful docstring.
- Private implementation details (e.g., `_internal_method`) do not necessarily need docstrings unless complex, particularly short descriptions are acceptable.
- Do not add a separate docstring to `__init__` when the class docstring documents constructor parameters.
- Document actual existing behavior, not intended or proposed behavior.
- Keep documentation concise and non-redundant.

### Class Attributes & Inheritance

- Document public attributes, properties, and parameter-derived state in an `Attributes` section.
- Do not repeat entries in `Attributes` if they are already documented in the class's own `Parameters` section.
- **Inherited Attributes:** Subclasses **must include inherited public attributes and properties** in their `Attributes` section. Users inspecting a subclass docstring should see its full public attribute interface without needing to check parent classes.

Example of subclass attribute documentation:

```python
class DiscreteMMActionSpace(DiscreteSpace):
    """Discrete action space for Market Making.

    Parameters
    ----------
    low : float
        Lower bound of the action values.
    high : float
        Upper bound of the action values.

    Attributes
    ----------
    shape : tuple of int
        The shape of a single physical element (inherited from Space).
    dtype : torch.dtype
        The data type of the physical elements (inherited from Space).
    ndim : int
        Number of dimensions of a single element (inherited from Space).
    num_elements : int
        Total number of valid discrete actions (inherited from DiscreteSpace).
    values : torch.Tensor
        Tensor containing all valid action pairs (inherited from DiscreteSpace).
    """
```

### Docstring Sections

Use standard NumPy section headers when applicable:

- `Parameters`
- `Returns`
- `Yields`
- `Raises`
- `Attributes`
- `Examples`
- `Notes`
- `See Also`

#### Parameters & Type Conventions

- For optional parameters, append `, optional` to the type. **Do not explicitly write `None`** in the type description (e.g., use `float, optional` rather than `float or None` or `float | None`).
- Do not mention default values in parameter descriptions unless the default value or its functional behavior is non-obvious, complex, or contextually critical.

```python
Parameters
----------
low : float
    Lowest price allowed on the grid.
min_spread : float, optional
    Minimum admissible value for ask - bid. If omitted, defaults to the tick size.
device : torch.device, optional
    Device on which the tensor is stored.
```

#### Returns Section

- Naming the return variable (e.g., `mask : torch.Tensor`) is optional. Include a name only if it clarifies the purpose or helps the user understand the return value, in particular when multiple return values are present.
- If no appropriate name exists or if the returned object is self-explanatory/trivial, specify only the type.

```python
Returns
-------
torch.Tensor
    Boolean mask tensor indicating validity for each sample.

Returns
-------
indices : torch.Tensor
    1D tensor containing the mapped discrete indices.
```

### Code references (plain text, no Sphinx markup)

Within docstrings, refer to code elements as plain text, exactly as they appear in the code (i.e., as imported or defined) — no backticks, double backticks, or Sphinx roles.

- Do **not** use double backticks for variables, attributes, literals, or values: write `self.size` or `None`, not ``self.size`` or ``None``.
- Do **not** use `:class:`, `:meth:`, or `:func:` roles: write `Space`, `DiscreteSpace.from_indices`, or `torch.clamp` as plain text, not `:class:`Space``, `:meth:`DiscreteSpace.from_indices``, or `:func:`torch.clamp``.
- Types, classes, methods, and functions are all written the same way: as plain dotted names (e.g., `torch.device`, `DiscreteSpace`, `numpy.ndarray`), matching how they'd be referenced in code.

---

## Comments

Review and refine inline comments in the code.

- Keep comments concise, accurate, and purposeful.
- Explain **why** something is done (intent/math/design reason), or clarify a step that isn't obvious from reading the code, rather than restating **what** the code does line by line.
- Use comments to divide a function, method, or class body into logical sections when it contains multiple distinct blocks of work (e.g., `# Setup defaults`, `# Axes setup`, `# Plotting loop`, `# Deduplicate legend and finalize`). This is especially useful in longer functions where the structure isn't otherwise obvious at a glance.
- Remove redundant, obvious, or outdated comments.
- Do not add comments merely to increase documentation lines.
- Comments must accurately reflect the code. Do not alter executable code to match a comment.
- Style: start each comment with a capital letter; do not end it with a period.

---

## Final review checklist

Before presenting the output, verify:

- [ ] **Executable code is 100% untouched** (no logic, variable name, or type hint changes).
- [ ] Public classes, methods, and functions have clean NumPy-style docstrings.
- [ ] `__init__` methods do not have redundant separate docstrings.
- [ ] Class `Attributes` sections list all public attributes, including **inherited attributes**.
- [ ] Docstrings do not duplicate `Parameters` in the `Attributes` section for the same class.
- [ ] Optional parameters use `, optional` without explicitly stating `None` in the type notation.
- [ ] Default values in docstrings are omitted unless non-obvious or vital for context.
- [ ] The `Returns` section uses names only when helpful; trivial/self-explanatory returns use type-only notation.
- [ ] Code references are plain text (e.g., `torch.device`, `Space`) with **no** double backticks and **no** Sphinx roles (`:class:`, `:meth:`, `:func:`).
- [ ] Inline comments explain "why" or clarify non-obvious steps instead of restating "what", and noise has been removed.
- [ ] Longer functions/classes with distinct logical blocks use section-dividing comments.
- [ ] Comments start with a capital letter and have no trailing period.