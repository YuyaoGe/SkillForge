"""Run one local, deterministic SkillForge lifecycle cycle."""

from pathlib import Path

from skillforge import EvolutionConfig, ForgeEngine, SkillBank


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bank = SkillBank.from_json(root / "data/skill_banks/alfworld.json")
    engine = ForgeEngine(
        EvolutionConfig(
            retire_min_usage=4,
            original_skill_protection_usage=4,
            stabilize_min_usage=4,
            promote_min_usage=2,
            max_total_skills=len(bank) + 1,
        )
    )

    retrieved = bank.retrieve("put the object in the receptacle", top_k=3)
    references = bank.references(retrieved)
    engine.record_episode(bank, references, success=True)
    engine.record_episode(bank, references, success=False)

    summary = engine.maybe_evolve(bank, global_step=1)
    print("retrieved:", [skill["skill_id"] for skill in references])
    print("counts:", bank.counts())
    print("cycle:", summary)


if __name__ == "__main__":
    main()
