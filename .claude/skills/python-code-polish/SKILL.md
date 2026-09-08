# Python Code Polish

Polish existing Python code for readability, typing, documentation, comments, and consistency **without changing its logic or behavior**.

## Critical constraint: preserve behavior

This is the highest-priority rule.

Do not change:

- algorithms, control flow, conditions, ordering, side effects, or data transformations;
- APIs, public interfaces, defaults, or exception types;
- variable, class, function, method, attribute, or parameter names;
- functionality or validation behavior.

Do not fix bugs, add validation, add error handling, or perform unrelated refactoring.

Only make changes related to:

- type hints;
- docstrings;
- comments;
- exception message wording;
- explicit final `return` statements;
- formatting required by those changes.

When in doubt, preserve the existing code.

---

## Type hints

Review type hints for consistency and completeness.

- Functions and methods should have explicit parameter type hints and return types.
- `__init__` should use `-> None`.
- Use precise types when they can be determined reliably from the existing code.
- Follow the Python version and typing conventions already used by the project.
- Do not introduce `Any` or speculative types unnecessarily.
- Do not change runtime behavior to satisfy typing.

Do not make type-hint changes when the intended type cannot be determined confidently.

---

## Return statements

Every function and method should have an explicit return type.

When a function or method does not return a value, use `-> None` and add an explicit `return` at the end:

```python
def process_data(data: list[str]) -> None:
    ...
    return
```

This rule applies **only to the final return statement**. Do not add, remove, move, or modify other `return` statements, as that could change control flow or behavior.

If the function already ends with an explicit `return`, preserve it.

---

## Exception messages

Review existing exception messages in the modified code.

Messages should:

- start with an uppercase letter;
- end with a period;
- clearly and concisely describe the problem;
- preserve useful context.

If a code identifier appears at the beginning of the message, preserve its original spelling/casing and surround it with single quotes:

```python
raise ValueError("'capacity' must be greater than 0.")
```

For interpolated values, use an f-string:

```python
raise ValueError(
    f"Tensor shape {x.shape} does not match buffer element shape {self.shape}."
)
```

Only modify the text of existing exceptions. Never change exception types, conditions, or semantics.

---

## Docstrings

Use concise **NumPy-style docstrings**.

### General rules

- Public classes, functions, and methods should have appropriate docstrings.
- Private implementation details do not necessarily need docstrings.
- Do not add a separate docstring to `__init__` when the class docstring documents its constructor parameters.
- Document existing behavior, not intended or corrected behavior.
- Avoid redundant or unnecessarily verbose documentation.
- Do not document implementation details unless they are relevant to the API.

### Sections

Use only relevant sections:

- `Parameters`
- `Returns`
- `Yields`
- `Raises`
- `Attributes`
- `Examples`
- `Notes`
- `See Also`
- `References`

Use standard NumPy formatting:

```python
Parameters
----------
name : type
    Description.

Returns
-------
type
    Description.

Raises
------
ValueError
    Description of when the exception is raised.
```

For optional parameters, use `optional` without unnecessarily documenting the default value:

```python
device : torch.device, optional
    Device on which the data is stored.
```

Document constructor parameters in the class docstring when appropriate.

### Code references

In docstrings:

- variables, attributes, constants, literals, values, and expressions use double backticks: ``self.size``;
- classes use `:class:`;
- methods use `:meth:`;
- functions use `:func:`.

For example:

```python
The buffer is initialized using :func:`torch.get_default_device`.

The method behaves consistently with :meth:`RingBuffer.get_all`.

If ``self.size`` is zero, the buffer is empty.
```

---

## Comments

Review comments in the modified code.

- Keep comments concise and purposeful.
- Improve existing comments when they are unclear, inaccurate, or unnecessarily verbose.
- Add comments only when they provide meaningful context that is not obvious from the code.
- Prefer explaining **why** something is done rather than restating **what** the code does.
- Remove redundant comments when they add no useful information.
- Do not add comments merely to increase documentation coverage.

Comment only when the comment adds clarity that the code itself cannot reasonably provide.

Comments must accurately describe the existing behavior. Do not modify code to make a comment true.

---

## Formatting

Keep formatting consistent with the existing project.

Do not reformat unrelated code. Only change formatting when required by the modifications above.

---

## Final review

Before finishing, verify:

- [ ] Logic and behavior are unchanged.
- [ ] APIs, defaults, names, and exception types are unchanged.
- [ ] No unrelated refactoring was performed.
- [ ] Type hints are complete and reliable.
- [ ] Functions and methods have explicit return types.
- [ ] Functions and methods that return nothing have a final explicit `return`.
- [ ] Existing non-final `return` statements were not changed.
- [ ] Exception messages start with uppercase letters and end with periods.
- [ ] Identifiers at the beginning of exception messages preserve their spelling and use single quotes.
- [ ] Interpolated exception messages use f-strings.
- [ ] Public APIs have appropriate NumPy-style docstrings.
- [ ] `__init__` has no redundant docstring.
- [ ] Docstring code references use the appropriate backtick/Sphinx syntax.
- [ ] Comments are concise, accurate, and genuinely useful.
- [ ] No unrelated formatting changes were made.
- [ ] The final diff contains only changes relevant to this skill.
