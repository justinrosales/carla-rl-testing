# Walkthrough: Automatic Graph Generation in Benchmark Framework

## Changes Made

Added ~240 lines to [benchmark.py](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py) with four new plotting functions and a CLI flag:

| Function | What It Plots | Output File |
|---|---|---|
| [plot_rewards()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#406-461) | Mean reward bar chart + per-episode grouped bars | `reward_comparison.png`, `reward_per_episode.png` |
| [plot_loss()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#491-546) | Training loss curves from TensorBoard event files | `loss_curves.png` |
| [plot_policy_comparison()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#552-604) | Episode length bars + efficiency scatter | `episode_length_comparison.png`, `efficiency_scatter.png` |
| [generate_all_graphs()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#610-623) | Orchestrator — calls all three above | — |

All PNGs save to `<output_dir>/graphs/`. Use `--no-graphs` to skip.

## Verification

Ran mock-data test ([/tmp/test_plots.py](file:///tmp/test_plots.py)) — all 4 expected PNGs generated successfully, loss curve gracefully skipped when no TensorBoard data was present.

```
reward_comparison.png                    OK (31,480 bytes)
reward_per_episode.png                   OK (29,183 bytes)
episode_length_comparison.png            OK (42,149 bytes)
efficiency_scatter.png                   OK (37,275 bytes)
loss_curves.png                          SKIPPED (expected — no TB data)

ALL TESTS PASSED
```

CLI `--help` confirmed `--no-graphs` flag is visible and documented.
