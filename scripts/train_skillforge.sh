#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SKILLFORGE_CONFIG:-}" && -f "$SKILLFORGE_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$SKILLFORGE_CONFIG"
fi

# Public launcher for the optional verl integration. All paths and provider
# settings are supplied by the caller; no private endpoint or machine path is
# embedded in this script.
: "${MODEL_PATH:?Set MODEL_PATH to a local model checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a training dataset}"
: "${VAL_FILE:?Set VAL_FILE to a validation dataset}"
: "${SKILL_BANK:?Set SKILL_BANK to a skill-bank JSON file}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/skillforge}"
ENV_NAME="${ENV_NAME:-alfworld/AlfredTWEnv}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  env.env_name="$ENV_NAME" \
  +env.use_skills_only_memory=True \
  +env.skills_only_memory.skills_json_path="$SKILL_BANK" \
  +env.skills_only_memory.retrieval_mode=template \
  +env.skills_only_memory.top_k=6 \
  +env.skills_only_memory.enable_dynamic_update=True \
  +env.skills_only_memory.enable_evolution=True \
  +env.skills_only_memory.evolution.evolve_every_n_validations=1 \
  +env.skills_only_memory.evolution.max_total_skills=80 \
  +env.skills_only_memory.evolution.max_mutations_per_step=2 \
  +env.skills_only_memory.evolution.max_retirements_per_step=3 \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$OUTPUT_DIR"
