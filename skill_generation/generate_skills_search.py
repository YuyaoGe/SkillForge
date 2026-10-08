"""Generate a public Search skill bank from trajectory memory using the configured teacher."""

import argparse
import json
from pathlib import Path
try:
    from skill_generation.client import ConfiguredTeacher
except ModuleNotFoundError:  # direct execution from a source checkout
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skill_generation.client import ConfiguredTeacher
from typing import List, Dict, Any



def load_memories(json_path: str) -> List[Dict]:
    """Load existing memory data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def classify_query_type(mem: Dict) -> str:
    """Classify a memory into a query type based on data source and question structure."""
    data_source = mem['tags'].get('data_source', '')
    goal = mem['content']['task_meta']['original_goal'].lower()

    if data_source in ('hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'):
        if any(kw in goal for kw in ['both', 'are the', 'which of', 'same', 'common', 'more', 'less', 'older', 'younger', 'taller', 'shorter']):
            return 'comparison'
        else:
            return 'multi_hop_reasoning'

    if data_source == 'popqa':
        return 'entity_attribute_lookup'

    return 'direct_retrieval'


def categorize_by_query_type(memories: List[Dict]) -> Dict[str, Dict[str, List]]:
    """Categorize memories by query type and outcome."""
    categorized = {
        'direct_retrieval': {'success': [], 'failure': []},
        'multi_hop_reasoning': {'success': [], 'failure': []},
        'entity_attribute_lookup': {'success': [], 'failure': []},
        'comparison': {'success': [], 'failure': []},
    }

    for mem in memories:
        query_type = classify_query_type(mem)
        outcome = 'success' if mem['tags']['outcome'] == 'Success' else 'failure'
        categorized[query_type][outcome].append(mem)

    return categorized


def extract_patterns(memories: List[Dict]) -> str:
    """Extract key patterns from memories for prompt."""
    patterns = []
    for mem in memories[:10]:  # Limit to 10 for context
        goal = mem['content']['task_meta']['original_goal']
        data_source = mem['tags'].get('data_source', '')
        trajectory = mem['content'].get('refined_trajectory') or []
        if isinstance(trajectory, dict):
            trajectory = trajectory.get('refined_trajectory', [])
        strategic = mem['content'].get('strategic_guidelines') or {}
        if 'strategic_guidelines' in strategic:
            strategic = strategic['strategic_guidelines']
        mistakes = strategic.get('mistakes_to_avoid', []) if strategic else []
        planning = strategic.get('planning_pattern', '') if strategic else ''

        pattern = {
            'goal': goal,
            'data_source': data_source,
            'steps': [{'action': s.get('action', ''), 'reasoning': s.get('reasoning', '')}
                     for s in trajectory[:5]] if isinstance(trajectory, list) else [],
            'planning_pattern': planning,
            'mistakes': mistakes[:3] if mistakes else []
        }
        patterns.append(pattern)

    return json.dumps(patterns, indent=2)


def generate_general_skills(client: ConfiguredTeacher, categorized_memories: Dict) -> List[Dict]:
    """Generate general skills using the configured teacher."""

    all_successes = []
    all_failures = []
    for query_type, data in categorized_memories.items():
        all_successes.extend(data['success'][:5])
        all_failures.extend(data['failure'][:5])

    success_patterns = extract_patterns(all_successes)
    failure_patterns = extract_patterns(all_failures)

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Analyze these successful and failed trajectories from a Search AI agent that answers questions by issuing search queries and reading retrieved documents.

The agent operates in a search environment where it can:
- Issue search queries to retrieve relevant documents
- Read and analyze retrieved documents
- Formulate answers based on evidence found

The agent handles various question types across datasets:
- Direct factoid retrieval (Natural Questions, TriviaQA)
- Entity attribute lookup (PopQA)
- Multi-hop reasoning (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle)

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 8-12 GENERAL SKILLS that apply across ALL question types. These should be:
1. **Concise** - Each skill should be 1-2 sentences max
2. **Actionable** - Clear what to do, not vague principles
3. **Transferable** - Apply to direct retrieval, multi-hop, comparison, and entity lookup tasks
4. **Failure-aware** - Derived from what went wrong in failures

Format as JSON array:
[
    {{
        "skill_id": "gen_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight in 1-2 sentences",
        "when_to_apply": "Specific trigger condition"
    }}
]

Focus on:
- Query formulation strategies (how to construct effective search queries)
- Evidence extraction and verification
- Multi-step decomposition for complex questions
- Handling ambiguous entities or questions
- Knowing when to refine vs. when to answer
- Avoiding hallucination (answering without evidence)

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_query_type_skills(client: ConfiguredTeacher, query_type: str,
                                successes: List[Dict], failures: List[Dict]) -> List[Dict]:
    """Generate query-type-specific skills."""

    if not successes and not failures:
        return []

    success_patterns = extract_patterns(successes[:8])
    failure_patterns = extract_patterns(failures[:8]) if failures else "[]"

    type_descriptions = {
        'direct_retrieval': 'Answer factoid questions (who/what/when/where) by searching and extracting answers directly from documents. Sources: Natural Questions, TriviaQA.',
        'multi_hop_reasoning': 'Answer questions requiring chained reasoning across multiple entities or facts. Must decompose the question, search for intermediate facts, and combine them. Sources: HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle.',
        'entity_attribute_lookup': 'Look up specific attributes (occupation, birthplace, genre, etc.) of named entities. Sources: PopQA.',
        'comparison': 'Compare two or more entities on a specific attribute (e.g., same nationality, both in same city). Requires retrieving info about each entity then synthesizing. Sources: HotpotQA, 2WikiMultiHopQA.',
    }

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable skills.

Query Type: {query_type.upper().replace('_', ' ')}
Description: {type_descriptions.get(query_type, '')}

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 4-6 QUERY-TYPE-SPECIFIC SKILLS for {query_type} tasks. These should be:
1. **Concise** - 1-2 sentences max per skill
2. **Specific** - Apply specifically to {query_type} questions
3. **Actionable** - Clear steps or decision rules
4. **Pattern-based** - Identify what makes success vs failure

Format as JSON array:
[
    {{
        "skill_id": "{query_type[:3]}_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight",
        "when_to_apply": "Specific trigger condition"
    }}
]

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_common_mistakes(client: ConfiguredTeacher, categorized_memories: Dict) -> List[Dict]:
    """Generate common mistakes to avoid."""

    all_failures = []
    for query_type, data in categorized_memories.items():
        for mem in data['failure'][:5]:
            sg = mem['content'].get('strategic_guidelines', {})
            if 'strategic_guidelines' in sg:
                sg = sg['strategic_guidelines']
            mistakes = sg.get('mistakes_to_avoid', [])
            if mistakes:
                all_failures.append({
                    'query_type': query_type,
                    'goal': mem['content']['task_meta']['original_goal'],
                    'data_source': mem['tags'].get('data_source', ''),
                    'description': mem.get('contextual_description', '')[:200],
                    'mistakes': mistakes[:3]
                })

    failure_data = json.dumps(all_failures[:20], indent=2)

    prompt = f"""You are an expert at analyzing agent failures and distilling them into avoidable mistakes.

Analyze these failure patterns from a Search AI agent that answers questions by issuing search queries and reading retrieved documents:

{failure_data}

Generate 8-12 COMMON MISTAKES to avoid. Format as JSON array:
[
    {{
        "mistake_id": "err_001",
        "description": "What the mistake is (1 sentence)",
        "why_it_happens": "Why agents make this mistake (1 sentence)",
        "how_to_avoid": "Concrete actionable fix (1-2 sentences)"
    }}
]

Focus on:
- Query formulation errors (too vague, wrong entity, not decomposing multi-hop)
- Evidence handling failures (hallucinating without evidence, misreading documents)
- Ambiguous entity resolution failures
- Repeating the same ineffective query
- Failing to decompose complex questions into sub-questions
- Premature answering before gathering sufficient evidence
- Misinterpreting retrieved documents

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="trajectory JSON file")
    parser.add_argument("--output", required=True, help="output skill-bank JSON file")
    args = parser.parse_args(argv)
    memory_json_path = args.input
    output_path = args.output

    print("Loading existing memories...")
    memories = load_memories(memory_json_path)
    print(f"Loaded {len(memories)} memories")

    print("\nCategorizing by query type...")
    categorized = categorize_by_query_type(memories)
    for query_type, data in categorized.items():
        print(f"  {query_type}: {len(data['success'])} success, {len(data['failure'])} failure")

    client = ConfiguredTeacher()

    print("\n=== Generating General Skills ===")
    general_skills = generate_general_skills(client, categorized)
    print(f"Generated {len(general_skills)} general skills")

    print("\n=== Generating Query-Type-Specific Skills ===")
    query_type_skills = {}
    for query_type, data in categorized.items():
        print(f"  Processing {query_type}...")
        skills = generate_query_type_skills(
            client, query_type,
            data['success'], data['failure']
        )
        query_type_skills[query_type] = skills
        print(f"    Generated {len(skills)} skills")

    print("\n=== Generating Common Mistakes ===")
    common_mistakes = generate_common_mistakes(client, categorized)
    print(f"Generated {len(common_mistakes)} mistakes")

    output = {
        "general_skills": general_skills,
        "task_specific_skills": query_type_skills,
        "common_mistakes": common_mistakes,
        "metadata": {
            "source": "generated from Search agent trajectories using the configured teacher",
            "total_memories_analyzed": len(memories),
            "query_type_distribution": {
                query_type: {
                    "success": len(data['success']),
                    "failure": len(data['failure'])
                }
                for query_type, data in categorized.items()
            }
        }
    }

    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    print(f"General skills: {len(general_skills)}")
    print(f"Query-type-specific skills: {sum(len(s) for s in query_type_skills.values())}")
    print(f"Common mistakes: {len(common_mistakes)}")
    print(f"\nSaved to: {output_path}")

    print("\n=== Sample General Skills ===")
    for skill in general_skills[:3]:
        print(f"\n[{skill.get('skill_id', 'N/A')}] {skill.get('title', 'N/A')}")
        print(f"  {skill.get('principle', 'N/A')}")


if __name__ == "__main__":
    main()
