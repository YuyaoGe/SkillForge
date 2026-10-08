# Environment adapters

This directory contains the public adapters used by SkillForge. The adapters
share a small environment contract and keep benchmark-specific dependencies
optional:

- `env_package/alfworld` — text and visual ALFWorld tasks
- `env_package/webshop` — WebShop shopping tasks
- `env_package/search` — search and multi-hop question tasks
- `env_package/sokoban` — Sokoban tasks
- `env_package/gym_cards` — Gym Cards tasks
- `env_package/appworld` — AppWorld tasks (optional)

Install the dependencies required by the benchmark you want to run. The
repository does not download datasets, model checkpoints, or service
credentials. WebShop and AppWorld require their own upstream setup commands;
run those commands in a separate environment and configure the adapter with
local paths.

The CPU tests use `skillforge.environments.ToyEnvironment`, so importing the
core package does not require any benchmark or GPU dependency.
