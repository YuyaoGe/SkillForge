# Third-party notices

The top-level MIT license applies to original SkillForge files unless a file
states another license. The following directories contain adapted or vendored
components whose files retain their source license and copyright headers:

- `verl/` — the trainer and runtime components, including their Apache,
  BSD, and other file-level notices.
- `agent_system/environments/` — benchmark adapters and their upstream
  environment packages.
- `agent_system/memory/` — memory and retrieval components.
- `agent_system/multi_turn_rollout/`, `agent_system/reward_manager/`, and
  `gigpo/` — the multi-turn rollout, reward, and GiGPO algorithm components.
- `agent_system/environments/env_package/search/third_party/` — the search
  environment's separately attributed dependencies.

The file-level headers are authoritative. This repository does not include
private provider endpoints, company SDKs, credentials, datasets, or machine
paths. Optional trainer and benchmark dependencies remain caller-supplied; see
`README.md` and `pyproject.toml`.
