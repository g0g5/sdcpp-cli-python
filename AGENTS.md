# AGENTS.md

## Purpose
This file guides agentic coding tools working in this repo.
It summarizes how to build, lint, test, and follow code style.

## Repository snapshot
- Language: Python 3.7+
- Entry example: `main.py`
- Core API: `sdcli.py` (fluent wrapper around `bin/sd-cli`)
- Binaries: `bin/` (stable-diffusion.cpp CLI)
- Models: `models/` (local weights, large files)

## Build / Lint / Test
There is no formal build system or test runner checked in.
Use the commands below as the closest equivalents.

### Run the example (basic smoke test)
- `python main.py`

### Run the CLI directly (integration smoke test)
- `./bin/sd-cli --help`
- `./bin/sd-cli --model models/your_model.safetensors --prompt "test" --output ./output.png`

### Lint / format
- No linter or formatter is configured in this repo.
- If you add one, document it here and keep defaults minimal.

### Unit tests
- No unit tests are present.
- If you add tests, include how to run all tests and a single test.

### Single test pattern (if you add pytest)
- `pytest -k "test_name" path/to/test_file.py`

## Environment notes
- Windows users may need `.\bin\sd-cli.exe` instead of `./bin/sd-cli`.
- The CLI requires compatible model weights; expect large files in `models/`.
- This repo does not vendor a virtual environment or dependency lock file.

## Code style guidelines
Follow existing patterns in `sdcli.py` and `main.py`.
Keep changes minimal and consistent with the fluent API style.

### Formatting
- Use 4-space indentation.
- Use double quotes for strings.
- Keep lines reasonably short; prefer readability over strict limits.
- Align docstrings and comments with existing style.
- Do not add decorative comments; only explain non-obvious logic.

### Imports
- Use standard library imports only unless a new dependency is justified.
- One import per line, grouped by standard library first.
- Keep imports at the top of the file.

### Types
- Use Python type hints, as in `sdcli.py`.
- Prefer `Optional[...]` for nullable values.
- Use concrete container types in public APIs (`List`, `Tuple`, `Sequence`).
- Keep return type hints on all public methods.

### Naming conventions
- Classes: `PascalCase` (e.g., `SdCppCli`).
- Methods/variables: `snake_case`.
- Fluent setters: `set_*`, `enable_*`, `disable_*`, `add_*`.
- Internal helpers: leading underscore (e.g., `_set_value`).

### Fluent API rules
- Public setters return `self` to allow chaining.
- Keep method names aligned to CLI flags.
- Avoid breaking existing flags or ordering semantics.
- Use `_set_flag`, `_set_value`, `_set_list` for consistency.
- When adding repeatable flags, pass `allow_multiple=True`.

### Error handling
- Let `subprocess.run(..., check=True)` raise on failure by default.
- Do not swallow errors silently.
- If adding validation, raise `ValueError` with clear messages.
- Avoid catching broad exceptions unless re-raising with context.

### File and path handling
- Accept user-provided paths as strings.
- Do not auto-expand or normalize paths unless required by a feature.
- Keep default binary path `./bin/sd-cli` unless user opts out.

### CLI argument rules
- Store arguments as `(flag, value)` tuples as in `_args`.
- Preserve insertion order for deterministic output.
- Avoid mutating existing entries except via `_remove_flag`.

### Documentation updates
- Update `README.md` for user-facing changes.
- Update `docs/CLI_manual.md` only if upstream CLI docs change.
- Keep examples minimal and runnable on a clean setup.

## Repo-specific conventions
- No configuration for lint/test/build exists today.
- Keep new tooling lightweight and optional.
- Large binaries and models should remain out of source control.

## Suggested local workflow
- Edit `sdcli.py`, then run `python main.py` for a quick check.
- Use `SdCppCli().build_args()` for debugging CLI arguments.
- Prefer adding a small example in `main.py` over ad-hoc scripts.

## Adding new options
- Add a new method that mirrors the CLI flag name.
- Keep docstrings concise and action-oriented.
- Reuse existing helpers; do not duplicate parsing logic.
- If a flag is deprecated upstream, note it in the docstring.

## Testing guidance (future)
- If adding tests, prefer `pytest` and keep tests fast.
- Mock `subprocess.run` for unit tests of `SdCppCli`.
- Keep integration tests optional due to large model requirements.

## Security and secrets
- Do not commit model weights, API keys, or user data.
- Avoid logging full paths to private models in tests.

## Cursor / Copilot rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
- If these files are added later, summarize their rules here.

## File list (quick reference)
- `sdcli.py`: fluent API wrapper around stable-diffusion.cpp CLI.
- `main.py`: runnable example.
- `docs/CLI_manual.md`: CLI option reference.
- `bin/`: CLI binaries and DLLs.
- `models/`: local model files.

## Agent etiquette
- Keep changes focused and incremental.
- Prefer editing existing files over adding new ones.
- Avoid reformatting unrelated code.
- Preserve existing API behavior unless explicitly asked.

## When you are unsure
- Read `README.md` and `sdcli.py` first.
- Ask a single, precise question only if blocked.
- Default to minimal, reversible changes.

## Changelog expectations
- No changelog file is present.
- If you introduce a changelog, keep it brief and factual.

## End
- Keep this file updated as tooling evolves.
- Target ~150 lines; concise is better.
