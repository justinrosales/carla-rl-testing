# Add Graph Generation to RL Benchmark

- [x] Plan graph generation methods (rewards, loss, policies)
- [x] Implement plotting functions in [benchmark.py](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py)
  - [x] [plot_rewards()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#406-461) — bar chart + per-episode breakdown
  - [x] [plot_loss()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#491-546) — TensorBoard loss curve parsing
  - [x] [plot_policy_comparison()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#552-604) — episode lengths + efficiency scatter
  - [x] [generate_all_graphs()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/benchmark.py#610-623) — orchestrator
- [x] Add `--no-graphs` CLI flag
- [x] Wire up graph generation in [main()](file:///Users/justinrosales/Projects/aiea-lab/carla-rl-testing/run.py#12-82)
- [x] Verify with mock data test script
