# Contributing

Thanks for helping improve CNC Detection and Monitoring Dashboard. Changes
should be focused, explain their behavior, and preserve the safety and
operational boundaries documented in the README.

## Before you start

- For a substantial change, open an issue first to discuss its scope and
  expected behavior.
- Check existing issues and pull requests for related work.
- Do not include RTSP credentials, `.env` files, camera footage, event data,
  SQLite databases, model weights, or other private/site data in a commit.
- Keep changes focused; do not combine unrelated refactoring with a feature
  or fix.

## Development setup

Use a supported Python version (3.10–3.12 recommended) and install
dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend from `cnc_backend` with
`python -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000`, and the
dashboard from `cnc_frontend` with
`python -m uvicorn web.app:app --host 127.0.0.1 --port 8000`.
See the README for full setup, model, and camera configuration instructions.

## Making a change

1. Create a focused branch from the current default branch.
2. Keep changes consistent with the existing Python, FastAPI, Jinja, and
   JavaScript patterns.
3. Update the README or other documentation when behavior, configuration,
   API routes, or operational steps change.
4. Add or update automated tests when suitable tests exist. Otherwise, state
   the manual verification performed and any environment limitations.
5. Never claim an inference, camera, or hardware behavior has been verified
   unless it was actually tested with the required model and device.

## Checks

Before opening a pull request:

- Run `git diff --check`.
- Compile changed Python files with `python -m py_compile <changed-files>`.
- Run relevant tests if present.
- For dashboard or backend behavior changes, smoke-test the affected route or
  workflow where practical. Camera/model integration checks require a
  reachable stream and compatible model weights.
- Review the diff and ensure generated files, credentials, videos, databases,
  logs, and model weights are not included.

## Pull requests

Use the pull request template. Explain the problem, behavior change, and
verification. Link related issues and call out changes to safety rules,
event timing, data formats, configuration, or deployment requirements.

Contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
