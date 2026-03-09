These files consist of a CARLA gym environment, based on the [gym-carla](https://github.com/cjy1992/gym-carla.git) library. Currently, the gym environment is customized to provide observation state details using a front, rear, and 2 side cameras. A top down point-cloud based view is also made possible through LIDAR.

## Single-Algorithm (DQN) Benchmark
The run.py file runs a DQN algorithm from [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/), using the gym environment. Steps to use this environment:

1. Download and run the latest version of CARLA. This environment was tested on CARLA v0.9.15
2. Create and activate a conda environment. (This environment was tested on python 3.7)
3. Clone the repo and cd into the gym-carla folder.
Run the following:
4. `pip3 install -r requirements.txt`
5. `pip3 install -e .`
6. export python path to CARLA installation folder
7. Modify run.py with the port number you are running CARLA on, as well as any other parameters you would like to change.
8. python3 run.py

If all steps were successful, you should see a Pygame window visualizing the RL algorithm running.

## Multi-Algorithm Benchmark

The `benchmark.py` script provides a centralized "one-click" framework for training and comparing multiple RL algorithms on the CARLA environment. Supported algorithms: **DQN, A2C, ACER, PPO1, ACKTR, TRPO**.

### Quick Start (Nautilus)

```bash
# Terminal 1: Start CARLA (headless)
./CarlaUE4.sh -RenderOffScreen -opengl -benchmark -fps=20

# Terminal 2: Run the benchmark
python benchmark.py

# Terminal 3: Monitor via TensorBoard
tensorboard --logdir=./tensorboard/ --port=6006
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithms` | `all` | Comma-separated list (e.g. `DQN,A2C,PPO1`) |
| `--timesteps` | `15000` | Training timesteps per algorithm |
| `--eval-episodes` | `3` | Evaluation episodes per algorithm |
| `--port` | `2000` | CARLA server port |
| `--town` | `Town03` | CARLA town/map |
| `--output-dir` | `./benchmark_results` | Model/results save directory |
| `--tensorboard-dir` | `./tensorboard` | TensorBoard log directory |

### Output

- Trained models saved to `./benchmark_results/<ALGORITHM>/`
- Comparison summary printed to console
- Full results exported to `./benchmark_results/results.json`
- TensorBoard logs per algorithm at `./tensorboard/`
