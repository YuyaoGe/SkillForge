"""Run the complete SkillForge lifecycle on CPU with a deterministic environment."""

from skillforge import EvolutionConfig, ForgeEngine, SkillBank
from skillforge.environments import ToyEnvironment
from skillforge.training import SkillForgeLoop


bank = SkillBank.from_json("data/skill_banks/alfworld.json")
engine = ForgeEngine(EvolutionConfig(evolve_every_n_validations=1))


def policy(context: str) -> str:
    # A deterministic policy is enough to exercise the environment/trainer hooks.
    return context.split("Task: ", 1)[-1].split("\n", 1)[0]


result = SkillForgeLoop(
    bank,
    engine,
    ToyEnvironment(),
    policy,
).run(["pick up the object", "clean the object"], validations=2)

print({"episodes": result.episodes, "successes": result.successes})
print("skills:", bank.counts())
