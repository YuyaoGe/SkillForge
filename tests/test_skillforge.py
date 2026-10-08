import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from skillforge import EvolutionConfig, ForgeEngine, SkillBank, SkillMutator
from skillforge.environments import ToyEnvironment
from skillforge.integrations import SkillForgeHooks
from skillforge.training import SkillForgeLoop


def make_bank():
    return SkillBank(
        {
            "general_skills": [
                {
                    "skill_id": "seed_bad",
                    "title": "Bad",
                    "principle": "Avoid this",
                    "usage_count": 20,
                    "success_count": 0,
                },
                {
                    "skill_id": "seed_good",
                    "title": "Good",
                    "principle": "Do this",
                    "usage_count": 30,
                    "success_count": 27,
                },
                {
                    "skill_id": "seed_mid",
                    "title": "Mid",
                    "principle": "Improve this",
                    "usage_count": 10,
                    "success_count": 5,
                },
            ],
            "task_specific_skills": {},
            "common_mistakes": [],
        }
    )


class SkillForgeTests(unittest.TestCase):
    def test_arithmetic_evaluator_is_allow_list_only(self):
        import importlib.util

        module_path = Path("agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/safe_math.py")
        spec = importlib.util.spec_from_file_location("skillforge_safe_math", module_path)
        safe_math = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(safe_math)

        self.assertEqual(safe_math.evaluate_arithmetic("(6+2)*3"), 24)
        with self.assertRaises(safe_math.UnsafeExpression):
            safe_math.evaluate_arithmetic("__import__('os').getcwd()")

    def test_episode_credit_and_prompt(self):
        bank = make_bank()
        retrieved = bank.retrieve("anything", top_k=2)
        references = bank.references(retrieved)
        bank.record_episode(references, success=True, task="demo")
        self.assertTrue(references[0]["usage_count"] >= 21)
        self.assertIn("General principles", bank.format_prompt(retrieved))

    def test_lifecycle_transitions(self):
        bank = make_bank()
        engine = ForgeEngine(
            EvolutionConfig(
                retire_min_usage=10,
                original_skill_protection_usage=10,
                stabilize_min_usage=10,
                max_retirements_per_cycle=5,
                max_total_skills=10,
            )
        )
        summary = engine.maybe_evolve(bank, global_step=1)
        self.assertIn("seed_bad", [item["skill_id"] for item in summary["retired"]])
        self.assertEqual(bank.get_skill("seed_good")["lifecycle"], "stable")

    def test_mutation_starts_as_trial(self):
        bank = make_bank()
        engine = ForgeEngine(
            EvolutionConfig(
                retire_min_usage=100,
                original_skill_protection_usage=100,
                stabilize_min_usage=100,
                mutate_min_usage=5,
                max_mutations_per_cycle=1,
                max_total_skills=10,
            )
        )

        def fake_mutator(parent, _bank):
            return {
                "skill_id": "evo_001",
                "title": "Child",
                "principle": "A better rule",
                "when_to_apply": "When the parent fails",
            }

        summary = engine.maybe_evolve(bank, global_step=1, mutate_fn=fake_mutator)
        self.assertEqual(len(summary["mutated"]), 1)
        self.assertEqual(bank.get_skill("evo_001")["lifecycle"], "trial")
        self.assertEqual(bank.get_skill("evo_001")["parent_id"], "seed_mid")

    def test_stable_skill_is_demoted_before_retirement(self):
        bank = SkillBank(
            {
                "general_skills": [
                    {
                        "skill_id": "stable_seed",
                        "title": "Stable",
                        "principle": "A rule",
                        "usage_count": 20,
                        "success_count": 0,
                        "lifecycle": "stable",
                    }
                ],
                "task_specific_skills": {},
                "common_mistakes": [],
            }
        )
        engine = ForgeEngine(
            EvolutionConfig(
                retire_min_usage=10,
                original_skill_protection_usage=10,
                max_retirements_per_cycle=1,
            )
        )
        summary = engine.maybe_evolve(bank, global_step=1)
        self.assertIn("stable_seed", summary["demoted"])
        self.assertIn("stable_seed", [item["skill_id"] for item in summary["retired"]])

    def test_json_round_trip(self):
        bank = make_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            bank.save(path)
            loaded = SkillBank.from_json(path)
            self.assertEqual(loaded.counts(), bank.counts())

    def test_mutator_uses_next_numeric_id(self):
        bank = make_bank()
        bank.add({
            "skill_id": "evo_001",
            "title": "Existing child",
            "principle": "Keep this rule",
        })

        class FakeClient:
            def complete(self, _prompt):
                return '{"title": "Child", "principle": "A new rule"}'

        child = SkillMutator(client=FakeClient()).mutate(bank.all_skills()[0], bank)
        self.assertEqual(child["skill_id"], "evo_002")

    def test_capacity_removes_trial_skills_when_needed(self):
        bank = SkillBank(
            {
                "general_skills": [
                    {
                        "skill_id": "trial_a",
                        "title": "Trial",
                        "principle": "A rule",
                        "lifecycle": "trial",
                        "usage_count": 5,
                        "success_count": 0,
                    },
                    {
                        "skill_id": "stable_a",
                        "title": "Stable",
                        "principle": "A rule",
                        "lifecycle": "stable",
                        "usage_count": 20,
                        "success_count": 18,
                    },
                ],
                "task_specific_skills": {},
                "common_mistakes": [],
            }
        )
        summary = ForgeEngine(EvolutionConfig(max_total_skills=1)).maybe_evolve(
            bank, global_step=1
        )
        self.assertEqual(len(bank), 1)
        self.assertEqual(summary["retired"][0]["skill_id"], "trial_a")

    def test_cpu_training_loop_runs_end_to_end(self):
        bank = SkillBank(
            {
                "general_skills": [
                    {
                        "skill_id": "task_hint",
                        "title": "Mention the task",
                        "principle": "Include the task text in the action.",
                    }
                ],
                "task_specific_skills": {},
                "common_mistakes": [],
            }
        )
        engine = ForgeEngine(EvolutionConfig(evolve_every_n_validations=1))

        def policy(context):
            return context.split("Task: ", 1)[1].split("\n", 1)[0]

        result = SkillForgeLoop(
            bank,
            engine,
            ToyEnvironment(),
            policy,
        ).run(["alpha", "beta"], validations=2)

        self.assertEqual(result.episodes, 4)
        self.assertEqual(result.successes, 4)
        skill = bank.get_skill("task_hint")
        self.assertEqual(skill["usage_count"], 4)
        self.assertEqual(skill["success_count"], 4)
        self.assertEqual(len(result.validation_summaries), 2)

    def test_trainer_hooks_use_the_same_cpu_contract(self):
        bank = make_bank()
        engine = ForgeEngine(EvolutionConfig(retire_min_usage=100))
        hooks = SkillForgeHooks(bank, engine, top_k=1)
        context = hooks.before_episode("anything")
        hooks.after_episode(
            context,
            success=True,
            trajectory=[{"action": "finish"}],
        )
        summary = hooks.after_validation(global_step=1)
        self.assertEqual(context.references[0]["usage_count"], 21)
        self.assertIn("promoted", summary)

    def test_batch_mutation_from_failures_is_provider_neutral(self):
        class FakeClient:
            def complete(self, _prompt):
                return '[{"title": "Check state", "principle": "Verify state before acting.", "when_to_apply": "Before each action"}]'

        bank = make_bank()
        children = SkillMutator(client=FakeClient()).analyze_failures(
            [{"task": "demo", "trajectory": "failed action"}],
            bank,
        )
        self.assertEqual(children[0]["skill_id"], "dyn_001")
        self.assertEqual(children[0]["title"], "Check state")

    def test_public_teacher_generation_modules_import_without_network(self):
        for module in (
            "skill_generation.generate_skills_alfworld",
            "skill_generation.generate_skills_search",
            "skill_generation.generate_skills_webshop",
            "skill_generation.alfworld",
            "skill_generation.search",
            "skill_generation.webshop",
            "skill_generation.generate_kimi_skills",
            "skill_generation.generate_kimi_skills_webshop",
        ):
            __import__(module)

    def test_search_skill_bank_uses_shared_memory_schema(self):
        from agent_system.memory import SkillsOnlyMemory

        memory = SkillsOnlyMemory("data/skill_banks/search.json")
        self.assertIn("task_specific_skills", memory.skills)
        self.assertNotIn("query_type_skills", memory.skills)
        self.assertEqual(memory.retrieve("Which is older, Ada or Grace?")["task_type"], "comparison")

    def test_generation_cli_help_is_cpu_only(self):
        scripts = (
            "generate_skills_alfworld.py",
            "generate_skills_search.py",
            "generate_skills_webshop.py",
        )
        for script in scripts:
            completed = subprocess.run(
                [sys.executable, str(Path("skill_generation") / script), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--input", completed.stdout)
            self.assertIn("--output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
