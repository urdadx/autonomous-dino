# Autonomous dino

Simple endless runner built with `pygame`, plus a `NEAT` training loop that can learn to play the game.

## Demo: AI playing automously, no human control

<video src="https://github.com/urdadx/autonomous-dino/raw/main/.github/auto-dino-2026-04-26_14.35.42.mp4" controls muted playsinline width="100%"></video>


## Requirements

- Python `3.12+`
- `uv`

## Setup

Install dependencies into the project virtualenv:

```bash
uv sync
```

After that, Python packages will be available in `./.venv`.

## Project Layout

- `main.py`: manual playable game entrypoint
- `dino/`: core game logic, rendering, config, and environment wrapper
- `agents/`: manual agent, NEAT agent, and DQN-ready agent interface
- `training/train_neat.py`: NEAT training entrypoint
- `training/eval_agent.py`: watch a trained NEAT agent play
- `configs/neat.ini`: NEAT hyperparameters
- `models/`: saved winners and checkpoints

## Run The Game Manually

Start the local playable version:

```bash
python3 main.py
```

Controls:

- `Space` or `Up`: jump
- `R` or `Space` after game over: restart
- close the window: quit

## Watch A Trained Agent

Load the default saved winner and watch it play:

```bash
./.venv/bin/python -m training.eval_agent
```

Watch a specific saved model:

```bash
./.venv/bin/python -m training.eval_agent --model models/neat_winner.pkl
```

If you trained a separate single-jump model with a custom filename, point eval at that file instead:

```bash
./.venv/bin/python -m training.eval_agent --model models/neat_single_jump_winner.pkl
```

## NEAT Training

### Default Headless Training

Train without opening a preview window:

```bash
./.venv/bin/python -m training.train_neat
```

Default behavior:

- trains for up to `1000` generations
- uses `5` training episodes per genome
- uses `3` validation episodes for the best genome each generation
- stops early if the validated best score reaches `200`
- writes checkpoints every `5` generations
- writes the best winner to `models/neat_winner.pkl`

### Visual Training

Train and watch the best genome of each generation play in a preview window:

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 1000
```

Important details:

- the full population is still trained headlessly for speed
- the preview window shows the best genome from the current generation
- by default, each episode and preview run continues until the player dies
- only use `--max-seconds` or `--preview-max-seconds` if you explicitly want an early cutoff

### Longer Training Example

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 2000 --episodes 8 --validation-episodes 5
```

### Use Time Caps Only If You Want Them

This forces training and preview runs to end early even if the agent is still alive:

```bash
./.venv/bin/python -m training.train_neat --render --max-seconds 60 --preview-max-seconds 20
```

## Checkpoints And Resume Flow

Checkpoints are enabled in the current flow.

By default:

- a checkpoint is written every `5` generations
- checkpoint files are named like `models/neat-checkpoint-5`
- the latest best winner is written to `models/neat_winner.pkl`

Resume training from a checkpoint:

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 1000 --resume-checkpoint models/neat-checkpoint-50
```

Resume training and save checkpoints every generation:

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 1000 --resume-checkpoint models/neat-checkpoint-50 --checkpoint-every 1
```

Use custom checkpoint and winner output paths:

```bash
./.venv/bin/python -m training.train_neat --checkpoint-prefix models/neat-single-jump-checkpoint- --output models/neat_single_jump_winner.pkl
```

Important distinction:

- `models/neat_winner.pkl`: best single genome for evaluation/playback
- `models/neat-checkpoint-*`: full population snapshots used to continue training

## Recommended Training Workflow

### Fresh Training Run

1. Start training.
2. Let it write checkpoints.
3. Periodically evaluate the current winner.
4. Resume from a checkpoint if you want to continue improving the same population.

Example:

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 1000
./.venv/bin/python -m training.eval_agent --model models/neat_winner.pkl
```

### Single-Jump Fresh Run With Separate Output Files

If you want to keep outputs separate from older experiments:

```bash
./.venv/bin/python -m training.train_neat --render --target-score 200 --generations 1000 --checkpoint-prefix models/neat-single-jump-checkpoint- --output models/neat_single_jump_winner.pkl
```

Then evaluate it with:

```bash
./.venv/bin/python -m training.eval_agent --model models/neat_single_jump_winner.pkl
```

## Stopping Training Safely

If you interrupt training with `Ctrl+C`, the trainer will stop and keep the best genome seen so far.

If you are using `--render`, closing the preview window also interrupts training.

## Testing And Verification

### Syntax Check

Quickly verify that the project still parses:

```bash
python3 -m py_compile main.py dino/*.py agents/*.py training/*.py
```

### Minimal NEAT Smoke Test

Run a tiny training job to make sure the trainer works:

```bash
./.venv/bin/python -m training.train_neat --generations 1 --episodes 1 --validation-episodes 1 --max-seconds 1 --target-score 1 --checkpoint-every 1 --checkpoint-prefix /tmp/dino-neat-checkpoint- --output /tmp/dino-neat-winner.pkl
```

### Render-Path Smoke Test

Use a tiny render run to make sure preview mode works:

```bash
./.venv/bin/python -m training.train_neat --render --generations 1 --episodes 1 --validation-episodes 1 --max-seconds 1 --preview-max-seconds 0.2 --target-score 1 --checkpoint-every 1 --checkpoint-prefix /tmp/dino-neat-preview-checkpoint- --output /tmp/dino-neat-preview.pkl
```

### Manual Play Test

Run the game and make sure:

- the runner starts correctly
- `Space` jumps
- the player dies on obstacle collision
- `R` or `Space` restarts after death

Command:

```bash
python3 main.py
```

## Common Files

- `models/neat_winner.pkl`: default winner output
- `models/neat_single_jump_winner.pkl`: example custom winner output for single-jump training
- `models/neat-checkpoint-*`: default checkpoints
- `models/neat-single-jump-checkpoint-*`: example custom checkpoints
- `configs/neat.ini`: NEAT configuration

## Notes

- The game core is structured so a future DQN trainer can plug into the same environment.
- NEAT currently uses the compact numeric observation vector from `dino/game.py`.
- Training success is based on survival and obstacle clearing, not raw rendered pixels.
