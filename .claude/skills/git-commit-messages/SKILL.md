---
name: git-commit-messages
description: Write clear, accurate Git commit messages in Conventional Commits format based on the actual diff (staged or unstaged changes), never inventing intent, scope, or changes not present in the diff. Use when the user asks for a commit message, asks to describe staged/unstaged changes, or asks to commit changes with a proper message.
---

# Git Commit Messages

Create clear, concise, and accurate Git commit messages based on the actual changes in the working tree or staged diff.

## Core principle

The commit message must describe **what the diff actually changes**, not what might have been intended.

Never invent:

- changes that are not present in the diff;
- motivations that cannot be inferred reliably;
- issue numbers, ticket IDs, or references;
- breaking changes;
- tests or validation that were not actually performed.

When the intent is ambiguous, prefer a conservative description or ask for clarification.

---

## Inspect the changes

When Git is available, inspect the actual changes before writing the message. Do not rely only on the user's description.

1. Run `git status` to understand the repository state.
2. Run `git diff --cached` for staged changes and `git diff` for unstaged changes. Prioritize the one the user asked about; if unspecified and both exist, do not assume they belong to the same commit, and use `git status` to decide which changes to describe.
3. Read surrounding code when needed to understand a change.

If Git is unavailable, use the code changes provided in the current context.

Do not run `git add`, `git commit`, `git reset`, or any other command that modifies Git state unless the user explicitly asks.

---

## Format

```text
<type>[(scope)][!]: <description>

[body]

[footers]
```

The description is **mandatory**: never write a header with only a type and scope (e.g., `refactor(envs)`).

### Types

Choose the most specific type:

- `feat`: add or extend functionality;
- `fix`: fix incorrect behavior;
- `refactor`: restructure code without changing behavior;
- `perf`: improve performance (only with evidence);
- `docs`: documentation-only changes, including docstrings and comments (write `docs`, not `doc`);
- `test`: add or modify tests;
- `style`: formatting-only changes (whitespace, import order) with no content change;
- `build`: build system or dependency changes (e.g., `pyproject.toml`, environment files);
- `ci`: CI configuration changes;
- `chore`: maintenance that fits nothing above, including changes to tooling and agent config such as `.claude/` skills.

When a change spans several types, choose the one matching its primary purpose.

### Scope

The scope is optional. Use it only when it adds useful context and keeps the header short.

- Use the lowercase name of the affected package or module, matching existing directory or module names (e.g., `spaces`, `agents`, `plotting`, `storage`). Not class or function names.
- If the commit touches several areas, or the scope adds nothing, omit it. `fix: handle empty input` is fully valid.
- Do not invent a scope. Prefer omitting it over shortening the description.

### Subject line

- Imperative mood, lowercase after the prefix, no trailing period.
- At most 72 characters in total, prefix included.
- Describe the primary change, without unnecessary detail.

```text
fix(parser): handle empty input
```

not

```text
Fixed a bug in the parser when the input was empty.
```

### Breaking changes

Mark a breaking change only when the diff itself breaks the public API (removed or renamed public names, changed signatures or behavior). Add `!` after the type or scope (e.g., `refactor(agents)!: rename base module`) and, if it needs explaining, a `BREAKING CHANGE:` footer. Never mark one speculatively.

### Body

Add a body only when it provides context the subject cannot: why a non-obvious change was needed, behavioral consequences, design decisions, or trade-offs. Do not restate the diff.

- Leave one blank line after the subject.
- Wrap lines at about 72 characters.
- Keep it concise.

### Footers

Put footers (e.g., `BREAKING CHANGE:`, `Co-Authored-By:`) last, after a blank line. If the environment or the user requires a trailer, include it unchanged.

---

## Accuracy

The message must remain faithful to the diff. Do not claim that:

- tests were added if they were not;
- a bug was fixed if the change is only documentation or refactoring;
- performance improved without evidence;
- an API changed unless the diff changes it;
- behavior changed when only internal structure did.

---

## Multiple changes

If the diff contains unrelated changes, do not force them into one inaccurate description. Recommend splitting them into separate commits, indicating which files go in each. If a single commit is clearly intended, describe its primary purpose and mention secondary changes only when significant.

---

## Final review

- [ ] The message matches the relevant diff, with nothing invented (changes, motivations, references, breaking changes).
- [ ] The type is the most specific appropriate one (`docs` for docstring-only changes, `chore` for `.claude/` changes).
- [ ] The description is present, imperative, lowercase, without a trailing period, and the header is at most 72 characters.
- [ ] The scope is a lowercase package or module name, or omitted.
- [ ] The body is omitted unless it adds useful context; footers come last.
- [ ] No unrelated changes are implied, and no Git state was modified unless the user asked.
