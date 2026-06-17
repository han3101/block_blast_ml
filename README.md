# block-blast

Stage 1 headless Block Blast-style engine.

## Running tests

```bash
uv run pytest -q
```

## Running the server

```bash
uv run uvicorn app.server:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

The `--host 0.0.0.0` flag makes the server reachable over local wifi (e.g. from a Mac to a workstation).
