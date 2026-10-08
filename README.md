# 🚀 SkillForge: Co-Evolving Skills and Agents via Dynamic Skill Lifecycles

[![arXiv](https://img.shields.io/badge/arXiv-2610.09832-b31b1b?logo=arxiv)](https://arxiv.org/abs/2610.09832)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2610.09832)
[![Project Page](https://img.shields.io/badge/Project-Page-brightgreen)](https://geyuyao.com/publication/ge2026skillforge/)
[![SkillForge Search 7B](https://img.shields.io/badge/Hugging%20Face-Search%207B-yellow?logo=huggingface)](https://huggingface.co/YuyaoGe/SkillForge_Search_7B)
[![SkillForge WebShop 7B](https://img.shields.io/badge/Hugging%20Face-WebShop%207B-yellow?logo=huggingface)](https://huggingface.co/YuyaoGe/SkillForge_Webshop_7B)
[![SkillForge ALFWorld 7B](https://img.shields.io/badge/Hugging%20Face-ALFWorld%207B-yellow?logo=huggingface)](https://huggingface.co/YuyaoGe/SkillForge_Alfworld_7B)

<p align="center">
  <img src="https://geyuyao.com/skillforge/figures/teaser.webp" alt="SkillForge overview" width="96%">
</p>

SkillForge is a skill-lifecycle framework for reinforcement-learning agents.
It retrieves reusable skills before an episode, assigns credit after the
episode, and evolves the skill bank during validation. The public release
contains the core lifecycle, provider-neutral environment contracts, teacher
guided mutation, benchmark adapters, and a cleaned VERL integration.

## 📥 Model download

The released task-specific RL checkpoints are available from Hugging Face:

| Task | Download link |
| --- | --- |
| 🧭 ALFWorld | 🤗 [Hugging Face](https://huggingface.co/YuyaoGe/SkillForge_Alfworld_7B) |
| 🛍️ WebShop | 🤗 [Hugging Face](https://huggingface.co/YuyaoGe/SkillForge_Webshop_7B) |
| 🔍 Search | 🤗 [Hugging Face](https://huggingface.co/YuyaoGe/SkillForge_Search_7B) |

## 📁 Repository layout

```text
skillforge/                  Core bank, lifecycle, mutation, and integrations
skillforge/environments/     Provider-neutral environment contracts
agent_system/memory/         Trainer-compatible retrieval and evolution memory
agent_system/environments/   ALFWorld, WebShop, Search, and other adapters
verl/                        Bundled trainer, rollout, and reward components
skill_generation/            Public skill-bank generation entry points
data/skill_banks/            Example ALFWorld, Search, and WebShop banks
scripts/setup_environment.sh One-command virtual-environment setup
scripts/train_skillforge.sh  VERL training launcher
examples/                    Usage demonstrations
tests/                       Unit and integration tests
```

## 🛠️ Installation

For one-command setup, copy the configuration and run:

```bash
cp config/skillforge.env.example config/skillforge.env
bash scripts/setup_environment.sh config/skillforge.env
```

The setup script creates `.venv` and installs the selected extras. GPU and LLM
dependencies are opt-in. To validate the configuration without installing
anything:

```bash
bash scripts/setup_environment.sh --check config/skillforge.env
```

Keep API keys outside the configuration file.

## 🔥 Training with VERL

Install the optional training dependencies:

```bash
pip install -e '.[training]'
```

Then provide the model, datasets, and skill bank through the environment:

```bash
export MODEL_PATH=/path/to/model
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/validation.parquet
export SKILL_BANK=data/skill_banks/alfworld.json
export OUTPUT_DIR=outputs/skillforge
export ENV_NAME=alfworld/AlfredTWEnv
export N_GPUS_PER_NODE=1

bash scripts/train_skillforge.sh
```

The launcher enables skill retrieval, validation-time credit assignment,
evolution, and mutation hooks. It does not provide a model checkpoint or a
dataset, and it does not hide environment-specific services.

## 🧬 Teacher-guided mutation

Mutation can use any OpenAI-compatible or Anthropic-compatible teacher. The
provider, endpoint, model, and credential are read from the environment:

```bash
pip install -e '.[llm]'
export SKILLFORGE_LLM_BACKEND=openai
export SKILLFORGE_LLM_API_KEY=your_key
export SKILLFORGE_LLM_BASE_URL=https://your-public-endpoint/v1  # optional
export SKILLFORGE_LLM_MODEL=your-model
```

The core `SkillMutator` accepts an injected client for controlled experiments.
Generated skill banks can be created from trajectory-memory JSON:

```bash
python3 -m skill_generation.generate_skills_alfworld \
  --input path/to/alfworld_memories.json \
  --output outputs/alfworld.json

python3 -m skill_generation.generate_skills_search \
  --input path/to/search_memories.json \
  --output outputs/search.json

python3 -m skill_generation.generate_skills_webshop \
  --input path/to/webshop_memories.json \
  --output outputs/webshop.json
```

## 📊 Evaluation

The offline evaluator expects a Parquet file containing the configured
response, data-source, and reward-model columns:

```bash
pip install -e '.[training]'
export EVAL_DATA_PATH=/path/to/evaluation.parquet
python3 -m verl.trainer.main_eval ray_init.num_cpus=8
```

The evaluator computes the configured reward function per data source and
prints aggregate `test_score/<data_source>` values. Environment-based
validation uses the same reward and validation hooks during VERL training.

## ⚙️ Skill lifecycle

1. Seed skills receive a warmup fitness estimate.
2. Trial skills become active after enough uses.
3. Stable skills are demoted if their fitness drops.
4. Low-fitness active skills are retired after a usage threshold.
5. High-fitness active skills become stable.
6. Medium-fitness skills are sampled for mutation with inverse-fitness weights.
7. New children start as trial skills and the bank enforces a size cap.

`SkillBank` records usage, successes, parent-child relationships, and bounded
failure context. `EvolutionConfig` controls the thresholds and budgets.

## Citation

```perl
@misc{ge2026skillforgecoevolvingskillsagents,
      title={SkillForge: Co-Evolving Skills and Agents via Dynamic Skill Lifecycles},
      author={Yuyao Ge and Yiwei Wang and Yuchen He and Baolong Bi and Lingrui Mei and Jiayu Yao and Lizhe Chen and Shenghua Liu},
      year={2026},
      eprint={2610.09832},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.09832},
}
```

## Acknowledgments

This work builds on **[SKILLRL](https://github.com/aiming-lab/SkillRL)**,
**[VERL](https://github.com/verl-project/verl)**,
**[ALFWorld](https://github.com/alfworld/alfworld)**, and
**[WebShop](https://github.com/princeton-nlp/WebShop)**. We thank the authors
and contributors of these projects for making their work available to the
community.

The SkillForge code is distributed under the repository license. Bundled VERL,
environment, and reward components retain their upstream license headers and
notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
redistributing modified copies.
