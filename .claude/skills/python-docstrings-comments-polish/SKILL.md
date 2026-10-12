---
name: python-docstrings-comments-polish
description: Polish docstrings and inline comments in existing Python code (NumPy-style docstrings, Sphinx markup, purposeful comments) without changing any logic. Use when the user asks to improve, clean up, document, or comment existing Python code, or to review/fix docstrings, without altering behavior.
---

# Python Docstrings and Comments Polish

Polish existing Python code by improving **docstrings and inline comments** for clarity, accuracy, and completeness **without modifying any executable code or logic**.

## Critical Constraint: Preserve Code and Logic

This is the absolute highest-priority rule.
- **Do NOT touch any Python executable code:** Do not change variable names, control flow, functions, method signatures, return statements, type hints, exception types, or logic.
- Only add, modify, or remove **docstrings** and inline **comments**.
- When in doubt, preserve the existing code exactly as it is.

---

## Default Target

Apply the skill to the files or code the user points to (attached files, selections, or named paths).

If the user gives no target and does not say what to correct, apply it to the **Python files with uncommitted changes**: run `git status` and take every modified, staged, and untracked `.py` file (skip deleted files, notebooks, and non-Python files). Process each whole file, then list the files you touched. Never extend the scope to unchanged files.

---

## 1. Docstrings

Use concise, precise **NumPy-style docstrings**. Document actual existing behavior, not intended or proposed behavior.

### 1.1 General Guidelines
- **Coverage:** Public classes, functions, and methods must have appropriate docstrings. When you find empty docstring placeholders (e.g., `""""""`), fill them with meaningful content.
- **Private & Dunder objects:** Implementation details (e.g., `_internal_method`) and standard magic/dunder methods (e.g., `__len__`, `__getitem__`) only need at most short, single-line descriptions unless their logic is uniquely complex.
- **Classes vs. Constructors:** Document constructor parameters in the class-level docstring under the `Parameters` section. Do **not** add a separate docstring to the `__init__` method.
- **Inheritance:** Subclasses should include inherited public attributes in their `Attributes` section, but avoid repeating inherited methods unless they have unique behavior or parameters. In addition, don't add redundant docstrings for methods that are already documented in the parent class unless they have unique behavior or parameters.

### 1.2 Markup & Code References
Always use proper **Sphinx/reStructuredText markup** for code references and math equations.
- **Inline code & literals:** Use double backticks for variables, attributes, literals, or values (e.g., ` ``self.size`` `, ` ``None`` `, ` ``True`` `).
- **Cross-references (prose only):** In summaries and descriptions (including parameter/attribute/return descriptions), use standard Sphinx roles for classes, methods, and functions.
  - Class: `` :class:`~lama_lab.spaces.DiscreteSpace` ``
  - Method: `` :meth:`~lama_lab.agents.AgentExp3.estimate_signal` `` or `` :meth:`_act` ``
  - Function: `` :func:`torch.clamp` ``
- **Type fields (no roles):** In the type position of `Parameters`, `Attributes`, `Returns`, and `Yields`, write plain type names without Sphinx roles.
  - Project classes: short name (e.g., `Space`, `DiscreteSpace`, `Agent`), not the full import path.
  - External types: qualified name (e.g., `torch.Tensor`, `torch.dtype`), since a bare name would be ambiguous.
  - Example: `action_space : DiscreteSpace`, not `` action_space : :class:`~lama_lab.spaces.DiscreteSpace` ``.
- **Math & Equations:** Keep math minimal. A docstring is not the place to derive or fully describe an algorithm; cite the paper or reference in the `References` section instead. Add a formula only when it genuinely clarifies what the code computes or how to use it (e.g., a key update rule, the meaning of a hyperparameter, a shape or normalization convention). Prefer one short formula over several, and plain words when they suffice. When a formula is warranted, use LaTeX formatting.
  - Inline math: `` :math:`\hat\ell_{t,a} = \ell_t \mathbf 1\{a_t = a\} / p_{t,a}` ``
  - Display math (block equations):
    ```rst
    .. math:: 
       \mathbb E[R_T] \le \frac{\ln K}{\eta} + \frac{\eta T K}{2}
    ```

### 1.3 Specific Sections Formatting

Use standard NumPy section headers (`Parameters`, `Returns`, `Yields`, `Raises`, `Attributes`, `Examples`, `Notes`, `References`, `See Also`) when applicable.

- **`References`:** The only place for papers, books, or other external sources behind an algorithm. List one entry per source (authors, title, venue, year, and a link or DOI if available). Do not cite sources in `Notes` or `See Also`.
- **`Notes`:** Implementation remarks, caveats, and (sparingly) a short formula. May mention a reference by number or name, but the full citation stays in `References`.
- **`See Also`:** Only related classes, methods, or functions of the code base or its dependencies, never papers.

#### Parameters
The type field mirrors the type hint, with accepted types joined by `or` (including `None`, written `or None`, never `| None` or `Optional[...]`). Append `, optional` if and only if the parameter has a default value. If `None` has a special meaning, state it in the description. Omit default values in descriptions unless non-obvious.

```python
"""
Parameters
----------
reward_range : tuple of float
    Lower and upper bound of the reward, used to normalize it.
eta : float, optional
    Learning rate controlling how aggressively the distribution updates.
seed : int or None, optional
    Seed for the random number generator. If ``None``, a random seed is used.
init_x : torch.Tensor or list[float]
    Initial position.
"""
```

#### Attributes & Inheritance
Document public attributes and parameter-derived state in an `Attributes` section.
- **Stored unchanged:** If an attribute is just a constructor parameter stored as-is (same type and meaning), document it only under `Parameters`. Do **not** repeat it in `Attributes`.
- **Transformed:** If the attribute differs from the parameter (e.g., converted to a tensor, normalized, validated, different type), list it in `Attributes` with its actual type and meaning.
- **Internal state:** Public attributes that are not constructor parameters (e.g., `weights`, counters, buffers) go only in `Attributes`.

**Inherited Attributes:** Subclasses **must** include inherited public attributes so users can see the full interface without checking parent classes.

```python
class DiscreteMMActionSpace(DiscreteSpace):
    """Discrete action space for Market Making.

    ... [Parameters section omitted for brevity] ...

    Attributes
    ----------
    shape : tuple of int
        The shape of a single physical element (inherited from :class:`Space`).
    dtype : torch.dtype
        The data type of the physical elements (inherited from :class:`Space`).
    num_elements : int
        Total number of valid discrete actions (inherited from :class:`DiscreteSpace`).
    """
```

#### Returns / Yields
Naming the return variable is optional. Include a name only if it clarifies the purpose or helps distinguish multiple return values. If trivial, specify only the type. The type mirrors the return hint (including `or None`); never use `, optional` for returns.

```python
"""
Returns
-------
probs : torch.Tensor or None
    Tensor of shape ``(n_episodes, n_arms)`` containing the probabilities, or ``None`` if the agent does not maintain an explicit policy.

Returns
-------
list[bool]
    Boolean mask tensor indicating validity for each sample.
"""
```

---

## 2. Inline Comments

Review and refine inline comments in the code.

- **Explain "Why", not "What":** Clarify the intent, mathematical reasoning, or design choices behind a block of code. Do not restate what the syntax already makes obvious.
- **Logical Grouping:** Use comments to visually divide long functions or classes into logical sections.
  ```python
  # Subtract max for numerical stability (prevents overflow in exp)
  w_max = torch.amax(w, dim=1, keepdim=True)
  exp_w = torch.exp(w - w_max)
  
  # Base exponential weights probabilities
  probs = exp_w / torch.sum(exp_w, dim=1, keepdim=True)
  ```
- **Cleanup:** Remove redundant, obvious, or outdated comments. Do not add comments merely to inflate documentation.
- **Style:** Start each inline comment with a capital letter. Do not end it with a period.

---

## 3. Final Review Checklist

Before presenting the output, verify:
- [ ] **Executable code is 100% untouched** (no logic, variable name, or type hint changes).
- [ ] Public classes, methods, and functions have clean NumPy-style docstrings.
- [ ] `__init__`, `__len__`, and similar methods follow the rules (no redundant docstrings for `__init__`, short docstrings for dunder methods).
- [ ] Code references in prose use proper **Sphinx roles** (``:class:``, ``:meth:``, ``:func:``) and **double backticks** (` ``var`` `).
- [ ] Type fields (`Parameters`, `Attributes`, `Returns`, `Yields`) use plain type names with no Sphinx roles.
- [ ] Math is minimal and only present where it clarifies behavior or usage; any formula uses ``:math:`` or ``.. math::`` blocks.
- [ ] Class `Attributes` sections list public attributes not already documented under `Parameters` (unless transformed), including **inherited attributes**.
- [ ] Type fields mirror the type hints (accepted types joined by `or`, including `or None`); `, optional` appears only on parameters with a default.
- [ ] The `Returns` section uses names only when helpful; trivial returns use type-only notation.
- [ ] Inline comments explain "why" or act as section dividers; noisy/obvious comments are removed.
- [ ] Comments start with a capital letter and have no trailing period.