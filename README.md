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


## Training on PPO

```bash
uv run python -m rl.train_ppo                                          # defaults
uv run python -m rl.train_ppo --config config/long_train.yaml --save-log
```

Any YAML value can be overridden on the CLI; the override wins (e.g.
`--config long_train.yaml --ent-coef 0.05` uses the YAML for everything except
`ent_coef`). With `--save-log`, console output is mirrored to
`runs/<run_id>/train.log` — no shell `> file.log` redirect needed.

### Common launch recipes

```bash
# Smoke test (fast, verify nothing's broken)
uv run python -m rl.train_ppo --total-timesteps 50000 --n-envs 8 --eval-interval 10

# Named run (stable run_id instead of timestamp-sha)
uv run python -m rl.train_ppo --config config/long_train.yaml \
  --run-id 2c_survival_shaping --save-log

# Tweak exploration / ablate return normalization
uv run python -m rl.train_ppo --config config/long_train.yaml --ent-coef 0.05 --run-id 2c_ent05 --save-log
uv run python -m rl.train_ppo --config config/long_train.yaml --no-normalize-returns --run-id 2c_nonorm --save-log

# Reward shaping (survival / board health)
uv run python -m rl.train_ppo --config config/long_train.yaml \
  --line-clear-bonus 0.5 --game-over-penalty 1.0 --run-id 2c_shaping --save-log

# Throughput: async vec envs (subprocess per env, escapes the GIL)
uv run python -m rl.train_ppo --config config/long_train.yaml --vec-env async --n-envs 16 --save-log

# Resume from a checkpoint
uv run python -m rl.train_ppo --config config/long_train.yaml \
  --resume runs/<run_id>/checkpoints/ckpt_<step>.pt --save-log

# Device / mixed precision (--device auto|cpu|cuda|mps)
uv run python -m rl.train_ppo --total-timesteps 50000 --device cpu
uv run python -m rl.train_ppo --config config/long_train.yaml --device cuda --amp --compile --save-log

# Seed sweep (reproducibility / variance check)
for s in 0 1 2; do
  uv run python -m rl.train_ppo --config config/long_train.yaml --seed $s --run-id 2c_seed$s --save-log
done

# Run detached in the background, then tail the log
nohup uv run python -m rl.train_ppo --config config/long_train.yaml \
  --run-id 2c_long --save-log > /dev/null 2>&1 &
tail -f runs/2c_long/train.log
```

`--env-mode` accepts `at_least_one` (default), `random`, or `solvable`.