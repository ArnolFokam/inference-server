# Guidelines

This repo is for practice. The user writes the logic themselves.

- Never generate code logic, tests included. Write structure (signatures, stubs, imports) plus comments, unless asked.

## Tests
- Every test gets a docstring.
- One behavior per test, no overlap.
- Use hand-picked inputs with a visible expected output.
- Use the seeded `rng` fixture for random data.
- Cover edge cases that guard real bugs.
- Write failing tests first, with a stub so the file imports.
