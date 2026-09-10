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

When Git is available, use Git to inspect the actual changes before writing the commit message.

1. Run `git status` to understand the current repository state.
2. Run `git diff` to inspect unstaged changes.
3. Run `git diff --cached` to inspect staged changes.
4. Determine which changes are relevant to the commit being described.
5. Consider the surrounding code when necessary to understand the change.

Do not rely only on the user's description when the actual diff is available.

When the user asks for a commit message for staged changes, prioritize `git diff --cached`.

When the user asks for a commit message for unstaged changes, prioritize `git diff`.

If both staged and unstaged changes exist, do not assume they belong to the same commit. Use `git status` and the relevant diff to determine which changes should be described.

If Git is unavailable, use the code changes provided in the current context.

Do not run `git add`, `git commit`, `git reset`, or other commands that modify Git state unless the user explicitly asks you to do so.

---

## Commit message format

Use **Conventional Commits**:

```text
<type>[optional scope]: <description>
```

Common types:

- `feat` — add or extend functionality;
- `fix` — fix incorrect behavior;
- `refactor` — restructure code without changing intended behavior;
- `docs` — documentation-only changes;
- `test` — add or modify tests;
- `chore` — maintenance changes that do not fit the categories above;
- `perf` — performance improvements;
- `style` — formatting or style-only changes.

Choose the most specific appropriate type.

### Scope

The scope is **optional**.

Use a scope only when:

- the affected component or area is clear;
- the scope adds meaningful information;
- it follows the project's existing naming conventions, when applicable;
- it keeps the subject concise.

Do not invent a scope.

If the scope does not add useful information, makes the subject unnecessarily long, or would make the commit title less clear, **omit it**.

A commit without a scope is completely valid:

```text
fix: handle empty input
```

Prefer a scoped commit when the scope adds useful context:

```text
fix(parser): handle empty input
```

Do not force a scope simply because one could be identified.

---

## Subject line

The subject should:

- be concise;
- describe the primary change;
- use the imperative mood;
- start with a lowercase letter after the conventional commit prefix;
- not end with a period;
- avoid unnecessary detail;
- remain reasonably short, especially when a scope is used.

Prefer:

```text
fix(parser): handle empty input
```

over:

```text
Fixed a bug in the parser when the input was empty.
```

If adding a scope makes the title unnecessarily long, omit the scope instead of shortening the description excessively.

---

## Body

Add a commit body only when it provides useful context that cannot be expressed clearly in the subject.

The body may explain:

- why a non-obvious change was necessary;
- important behavioral consequences;
- relevant design decisions;
- constraints or trade-offs.

Do not restate the diff line by line.

Keep the body concise.

---

## Accuracy

The commit message must remain faithful to the diff.

Do not claim that:

- tests were added if they were not;
- bugs were fixed if the change is only documentation or refactoring;
- performance was improved without evidence;
- an API was changed unless the diff actually changes it;
- behavior changed when the diff only changes internal structure.

When a change spans multiple categories, choose the type that best represents the primary purpose of the change.

---

## Multiple changes

If the diff contains several unrelated changes, do not force them into an inaccurate single description.

When appropriate, recommend splitting the changes into separate commits.

If a single commit is clearly intended, describe its primary purpose and mention secondary changes only when they are significant.

---

## Final review

Before finishing, verify:

- [ ] The message accurately reflects the relevant diff.
- [ ] No changes, motivations, or references were invented.
- [ ] The Conventional Commit type is appropriate.
- [ ] The scope is optional and used only when it adds meaningful information.
- [ ] The scope follows project conventions when applicable.
- [ ] The scope does not make the subject unnecessarily long.
- [ ] The subject is concise and imperative.
- [ ] The subject does not end with a period.
- [ ] The body is omitted unless it adds useful context.
- [ ] No unrelated changes are implied.
- [ ] If Git is available, the final message was checked against the relevant diff.
- [ ] No Git state was modified unless explicitly requested by the user.