# Add Automatic Graph Generation to RL Benchmark Framework

The current [benchmark.py](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py) saves results as JSON and prints text tables, but **has no graph/visualization generation**. We need to add functions that automatically produce publication-quality plots for rewards, loss, and policy distributions after training completes.

## Proposed Changes

### Visualization Module

#### [MODIFY] [benchmark.py](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py)

Add matplotlib import and three new plotting functions, plus a CLI flag to control graph generation:

1. **`plot_rewards(results, output_dir)`** — Generates two charts:
   - **Bar chart**: Mean reward per algorithm with std-deviation error bars (comparison across algorithms)
   - **Per-episode breakdown**: Grouped bar or line chart showing individual episode rewards for each algorithm

2. **`plot_loss(tb_dir, output_dir)`** — Reads TensorBoard event files from the `--tensorboard-dir` and plots:
   - Training loss curves over timesteps for each algorithm on the same axes
   - Uses `tensorflow.compat.v1` (already a dependency via `stable_baselines`/`tensorflow==1.15.0`) to parse the event files

3. **`plot_policy_comparison(results, output_dir)`** — Generates:
   - **Episode length comparison**: Bar chart of mean episode lengths per algorithm (a proxy for policy quality—longer episodes = better lane-following policy)
   - **Efficiency scatter plot**: Training wall-time vs. mean reward scatter to visualize algorithm efficiency

4. **`generate_all_graphs(results, output_dir, tb_dir)`** — Orchestrator that calls all three plot functions and saves a summary PNG grid

5. **CLI changes**: Add `--no-graphs` flag (graphs enabled by default). Call `generate_all_graphs()` at the end of [main()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#445-501).

All plots will be saved as PNG files in `<output_dir>/graphs/`.

> [!NOTE]
> `matplotlib` is already a dependency in [gym-carla/requirements.txt](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/gym-carla/requirements.txt). TensorFlow 1.15 is also already listed as a dependency, so we can parse TensorBoard event files without adding new packages.

> [!IMPORTANT]
> The loss plotting depends on TensorBoard event files being written during training. The framework already configures `tensorboard_log` for each model, so this should work. However, the exact scalar tag names may vary by algorithm — we'll need to search for common tags like `loss`, `value_loss`, `policy_loss`, etc.

## Verification Plan

### Manual Verification

Since this code requires a running CARLA simulator to actually train models, we can verify the plotting logic independently:

1. **Dry-run test**: Create a small test script (`/tmp/test_plots.py`) that generates **mock results data** (matching the JSON structure produced by [benchmark.py](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py)) and calls the new plotting functions directly. This verifies the plots render without errors and produce valid PNGs.
   - Run: `python /tmp/test_plots.py`
   - Expected: PNG files appear in a temp output directory, no matplotlib errors

2. **Visual inspection**: Open the generated PNGs to confirm charts are readable, properly labeled, and have correct data.

3. **CLI flag test**: Run `python benchmark.py --help` and verify the `--no-graphs` flag appears in the help output.
