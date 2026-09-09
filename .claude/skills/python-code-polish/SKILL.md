# Python Typing, Messages, and Code Structure Polish

Polish existing Python code by enforcing **complete type hints, standardized exception/log messages, explicit return statements, and clean abstract method structures** without changing runtime behavior or business logic.

## Critical constraint: preserve code logic and behavior

This is the highest-priority rule.

- Do **NOT** change algorithms, control flow, conditions, data transformations, or execution order.
- Do **NOT** change public API signatures, variable names, class names, function names, default argument values, or exception types.
- Do **NOT** fix bugs, add new validation logic, or rewrite algorithms.
- Only modify type hints, message wording/formatting, final `return` statements, and `pass` placeholders in abstract methods.

---

## Type Hints

Ensure complete and precise type annotations across all functions and methods.

- **Completeness:** All parameters and return values must have explicit type hints.
- **`__init__` Methods:** Must always be annotated with `-> None`.
- **Modern Syntax:** Use modern Python typing conventions (e.g., `list[str]`, `dict[str, int]`, `float | None`) matching the project's target Python version.
- **Precision:** Avoid `Any` or speculative types unless strictly necessary or dictated by external dynamic libraries.
- Do not modify runtime code or add runtime type casting just to satisfy a type checker.

---

## Explicit Return Statements

Every function and method must have a clear and explicit execution exit.

- **Explicit Return Types:** Every function/method must declare its return type (e.g., `-> int`, `-> torch.Tensor`, `-> None`).
- **Final `return` for `-> None`:** Functions and methods that return `None` (including `__init__`) **must end with an explicit `return` statement** on a new line at the very end of the function body:

```python
def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    self.shape = shape
    self.dtype = dtype
    return


def log_status(message: str) -> None:
    print(f"[INFO] {message}")
    return
```

- **Non-final returns:** Do NOT touch, move, or modify existing internal `return` statements used for early exits or conditional logic.

---

## Abstract Methods

Abstract methods defined with `@abstractmethod` must use `pass` as their body placeholder.

- If an abstract method includes a docstring, place `pass` on a new line immediately following the docstring:

```python
@abstractmethod
def contains(self, x: torch.Tensor) -> torch.Tensor:
    """Check if a batch of elements belongs to the space."""
    pass
```

- Do not use `Ellipsis` (`...`) or `raise NotImplementedError` in abstract base methods unless specifically required by an external framework pattern.

---

## Messages (Exceptions, Logging, and Prints)

Standardize all human-readable text strings in `raise` exceptions, `logger` calls, and `print` statements.

### Formatting Rules

1. **Capitalization:** Start every message with an uppercase letter.
2. **Punctuation:** End every message with a period (`.`).
3. **Identifiers:** Surround variable names, parameter names, or code identifiers with single quotes (e.g., `'tick_size'`). Preserve their exact casing.
4. **Interpolation:** Always use f-strings for string interpolation.

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
# Incorrect
print("processing batch")

# Correct
print("Processing batch.")
```

---

## Final Review Checklist

Before finishing, verify:

- [ ] **Runtime logic is 100% unchanged** (no algorithmic, flow, or API modifications).
- [ ] Every function and method has full parameter and return type hints.
- [ ] `__init__` methods are annotated with `-> None`.
- [ ] Every function/method ending with `-> None` ends with an explicit `return` statement.
- [ ] Existing internal non-final `return` statements were preserved untouched.
- [ ] All `@abstractmethod` bodies use `pass`.
- [ ] Exception, log, and print messages start with a capital letter and end with a period.
- [ ] Code identifiers inside error/log messages are enclosed in single quotes `'identifier'`.
- [ ] Dynamic string formatting uses f-strings consistently.