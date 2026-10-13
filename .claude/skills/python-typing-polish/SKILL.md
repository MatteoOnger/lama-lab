---
name: python-typing-polish
description: Polish existing Python code by enforcing complete type hints, standardized exception/log/print messages, explicit return statements, and clean abstract method structures, without changing runtime behavior or business logic. Use when the user asks to add/complete type hints, standardize error or log messages, add explicit returns, or clean up abstract method bodies in existing Python code.
---

# Python Typing, Messages, and Code Structure Polish

Polish existing Python code by enforcing **complete type hints, standardized exception/log messages, explicit return statements, and clean abstract method structures** without changing runtime behavior or business logic.

## Critical constraint: preserve code logic and behavior

This is the highest-priority rule.

- Do **NOT** change algorithms, control flow, conditions, data transformations, or execution order.
- Do **NOT** change public API signatures, variable names, class names, function names, default argument values, or exception types.
- Do **NOT** fix bugs, add new validation logic, or rewrite algorithms. If you notice a bug or a hint that contradicts the code (e.g., an `is None` check on an attribute hinted as non-optional), report it in your final summary instead of changing it.
- Only modify type hints, message wording/formatting, final `return` statements, and `pass` placeholders in abstract methods.

---

## Default target

Apply the skill to the files or code the user points to (attached files, selections, or named paths).

If the user gives no target and does not say what to correct, apply it to the **Python files with uncommitted changes**: run `git status` and take every modified, staged, and untracked `.py` file (skip deleted files, notebooks, and non-Python files). Process each whole file, then list the files you touched. Never extend the scope to unchanged files.

---

## Type hints

Type hints are the source of truth for types, including the type fields of docstrings.

- **Completeness:** Every parameter and return value must be annotated, except `self` and `cls`.
- **`__init__`:** Always annotated with `-> None`.
- **Modern syntax:** Use `list[str]`, `dict[str, int]`, `float | None` (never `List`, `Optional`, `Union`), matching the project's target Python version. Import abstract types such as `Callable` and `Iterable` from `collections.abc`.
- **Precision:** Parametrize generics (`dict[str, Any]`, not bare `dict`; `tuple[float, float]`, not `tuple`). Avoid `Any` unless strictly necessary or dictated by an external dynamic library.
- Do not modify runtime code or add runtime casts just to satisfy a type checker.

### Variables and attributes

Annotate at boundaries (signatures, attribute initializations that inference cannot type) and let inference handle the interior.

**Annotate:**

- **Empty containers and `None` placeholders:** `self.history: list[float] = []`, `self.cache: dict[str, int] = {}`, `self.model: Model | None = None`.
- **Attributes that change type later:** declare the union up front (e.g., `self.eta: float | None = None`).
- **Too-narrow inference:** state the intended type (e.g., `self.count: float = 0`, since `0` infers `int`).
- **Right-hand sides typed `Any` or untyped** (e.g., config builders, lookups in a `dict[str, Any]`): annotate to restore the precise type (e.g., `self.expert: Agent = build_from_config(cfg)`).
- **Class-level fields** of dataclasses, `NamedTuple`, and `Protocol`, where annotations are required.

**Do not annotate:**

- **Attributes assigned from a typed parameter** (`self.eta = eta`); they inherit its type.
- **Locals with obvious right-hand sides** (`n = len(xs)`, `total = 0.0`, `name = "x"`) or anything the checker already infers correctly.
- **To silence a checker.** Fix the expression's type at its source instead; never use a blanket annotation to hide an `Any`.

Never remove an existing annotation just because it is redundant.

---

## Return statements

- **Final `return` for `-> None`:** This is a deliberate house style. Every function and method that returns `None` (including `__init__`) **must end with an explicit bare `return`** on its own line at the end of the body. Do not remove it as "useless" because a linter suggests so.

```python
def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    self.shape = shape
    self.dtype = dtype
    return


def log_status(message: str) -> None:
    print(f"[INFO] {message}")
    return
```

- **Exceptions:** Abstract methods (see below), and functions whose body ends with `raise`, do not get a final `return`.
- **Non-final returns:** Do not touch, move, or modify existing internal `return` statements used for early exits or conditional logic.

---

## Abstract methods

Methods decorated with `@abstractmethod` use `pass` as their body, placed on its own line after the docstring if there is one. This is deliberate house style and takes precedence over the final-`return` rule.

```python
@abstractmethod
def contains(self, x: torch.Tensor) -> torch.Tensor:
    """Check if a batch of elements belongs to the space."""
    pass
```

Do not use `...` or `raise NotImplementedError` in abstract methods unless an external framework pattern requires it.

---

## Messages (exceptions, warnings, logging, prints)

Standardize human-readable strings in `raise` exceptions, `warnings.warn`, `assert` messages, `logger` calls, and `print` statements.

### Formatting rules

1. **Capitalization:** Start every message with an uppercase letter.
2. **Punctuation:** End every message with a period. For a message built from several concatenated string literals, the period goes at the end of the whole message, not of each fragment.
3. **Quotes:** Surround code identifiers with single quotes, preserving their exact casing. Code identifiers are variable, parameter, attribute, function, and class names (e.g., `'tick_size'`, `'reward'`, `'DiscreteSpace'`, `'_act()'`). When a class name appears in a message, it is quoted (e.g., `'BlumMansour'`). Algorithm names (e.g., Blum-Mansour) are valid in messages and are never quoted. Prefer the class name when writing a new message, since it uniquely identifies a specific object, but do **not** rewrite an existing message that uses an algorithm name correctly, only fix its capitalization, punctuation, and quoting.
4. **Interpolation:** Always use f-strings, including in `logger` calls. Convert `+` concatenation, `%` formatting, and `.format()` to f-strings.

### Log messages

Apply the rules above to a `logger` call only if it is a plain human-readable sentence. If it follows a structured or machine-parsed format (e.g., `event=order_filled qty=10`, templates with trailing placeholders, or a consistent lowercase/no-punctuation convention across nearby calls), leave it untouched.

### Examples

```python
# Incorrect
raise ValueError("delta must be positive, got " + str(delta))

# Correct
raise ValueError(f"'delta' must be strictly positive. Got {delta}.")
```

```python
# Incorrect
raise ValueError("high must be greater than low")

# Correct
raise ValueError(f"'high' must be greater than 'low'. Got {low} and {high}.")
```

```python
# Incorrect (algorithm name quoted)
raise ValueError("'Blum-Mansour' requires a discrete action space.")

# Correct (algorithm name, unquoted)
raise ValueError("Blum-Mansour requires a discrete action space.")

# Also correct (class name, quoted)
raise ValueError("'BlumMansour' requires a discrete 'action_space'.")
```

```python
# Incorrect
print("processing batch")

# Correct
print("Processing batch.")
```

---

## Final review checklist

- [ ] **Runtime logic is 100% unchanged**; suspected bugs were reported, not fixed.
- [ ] Every function and method has full parameter and return hints, with parametrized generics and modern syntax.
- [ ] Variables and attributes are annotated only where inference fails (empty containers, `None` placeholders, changing or too-narrow types, `Any` right-hand sides), never for obvious locals or typed parameters.
- [ ] `__init__` is annotated with `-> None`.
- [ ] Every `-> None` function ends with an explicit `return`, except abstract methods and bodies ending in `raise`; internal returns are untouched.
- [ ] All `@abstractmethod` bodies use `pass`.
- [ ] Messages start with a capital letter and end with a period (once, at the end of the whole message).
- [ ] Variable, parameter, and class names in messages are single-quoted; algorithm names are valid and never quoted (class names are preferred only in new messages).
- [ ] Structured log messages were left untouched; all other interpolation, including in `logger` calls, uses f-strings.
- [ ] If no target was given, only the Python files with uncommitted changes were processed, and they are listed in the summary.
