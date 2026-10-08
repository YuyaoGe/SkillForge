# Memory manager

The memory modules provide optional retrieval and structured-skill storage for
long-horizon environment rollouts. They are imported by the trainer only when a
memory-backed adapter is enabled.

`SkillsOnlyMemory` reads a skill-bank JSON file and retrieves general and
task-specific entries. `RetrievalMemory` can additionally index trajectory
memories with optional embedding dependencies. Both implementations accept
caller-provided paths and model settings; no dataset, endpoint, or credential
is embedded in this repository.
