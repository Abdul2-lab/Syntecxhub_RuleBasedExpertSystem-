# Rule-Based Expert System (Syntecxhub AI Internship — Project 2)

A forward-chaining expert system engine with certainty-factor reasoning,
a data-driven (JSON) knowledge base, structured logging, a small CLI, and
an automated test suite.

## Architecture

```
expert_system_project/
├── expert_system.py       # engine: Rule, KnowledgeBase, InferenceEngine, CLI
├── rules.json              # data-driven knowledge base (rules + symptom list)
├── test_expert_system.py   # unit tests (unittest)
└── README.md
```

The rules and the engine are deliberately decoupled: **`rules.json` holds
all domain knowledge** (which symptoms exist, what rules connect them, and
how reliable each rule is), while `expert_system.py` contains only the
generic reasoning logic. This means you can point `--rules` at a completely
different knowledge base (a different domain entirely) without changing a
single line of engine code.

## Key design choices

| Concept | Why it's here |
|---|---|
| **`KnowledgeBase` / `Rule` / `InferenceEngine` split** | Separates data (facts, rules) from behavior (forward chaining), which is standard expert-system architecture (production rules + working memory + inference engine). |
| **Certainty factors (0–1)** | Real-world evidence is rarely 100% certain. Each rule has its own reliability (`certainty_factor`), and a fired rule's conclusion certainty = `rule.certainty_factor × min(certainty of its conditions)` — a fuzzy AND, since a chain of reasoning is only as strong as its weakest link. |
| **MYCIN-style certainty combination** | If two different rules independently support the same conclusion, their certainties are combined with `CF_combined = CF_old + CF_new × (1 − CF_old)`, so confidence grows with more supporting evidence instead of one answer overwriting another. |
| **`logging` module instead of `print`** | Lets you dial verbosity (`--quiet`), and optionally persist the full reasoning trace to a file (`--log-file`) for later review/grading — a print-only script can't do either. |
| **JSON-driven rules** | Non-programmers (or a future you) can add new rules/domains by editing a data file, not the code. |
| **`argparse` CLI** | Makes the script scriptable/automatable (`--facts`, `--demo`, `--quiet`) instead of only interactive, which matters for testing and demos. |
| **Unit tests** | Verifies forward chaining, multi-step rule chaining, and certainty-factor math actually behave as intended — not just "it ran once and looked okay." |

## How to run

Requires Python 3.9+ (uses only the standard library — no installs needed).

```bash
# Interactive — answer y/n (optionally with a confidence, e.g. "y,0.7") for each symptom
python3 expert_system.py

# Non-interactive / batch, useful for quick checks or automation
python3 expert_system.py --facts fever,cough,body_ache

# Built-in demo fact set
python3 expert_system.py --demo

# Only show final conclusions, hide the step-by-step reasoning log
python3 expert_system.py --demo --quiet

# Also save the full reasoning trace to a file
python3 expert_system.py --demo --log-file run.log
```

### Example output

```
[FACT ADDED] 'fever' (certainty=1.00)
[FACT ADDED] 'cough' (certainty=1.00)
[FACT ADDED] 'body_ache' (certainty=1.00)
[RULE FIRED] R1: IF ['cough', 'fever'] -> THEN 'suspect_flu' (certainty=0.80) — Fever combined with cough is a classic flu indicator.
[RULE FIRED] R2: IF ['body_ache', 'suspect_flu'] -> THEN 'recommend_rest_and_fluids' (certainty=0.72) — Likely flu plus body ache -> standard rest/fluids advice.

--- Conclusions (fact: certainty) ---
 -> suspect_flu: 0.80
 -> recommend_rest_and_fluids: 0.72
```

## Running the tests

```bash
python3 -m unittest test_expert_system.py -v
```

9 tests cover: fact storage, certainty-factor combination, single-rule
firing, multi-step chaining, rules correctly *not* firing when conditions
aren't fully met, independent rule groups not cross-triggering, certainty
being limited by the weakest condition, and the engine terminating cleanly
without an infinite loop.

## Sample rule base (`rules.json`)

| Rule | Conditions | Conclusion | Rule CF |
|------|-----------|------------|---------|
| R1 | fever, cough | suspect_flu | 0.80 |
| R2 | suspect_flu, body_ache | recommend_rest_and_fluids | 0.90 |
| R3 | sneezing, runny_nose, itchy_eyes | suspect_allergy | 0.75 |
| R4 | suspect_allergy | recommend_antihistamine | 0.85 |
| R5 | suspect_flu, shortness_of_breath | recommend_see_doctor | 0.95 |
| R6 | fever, sore_throat | suspect_infection | 0.70 |

## Extending it

Add a new object to the `"rules"` array in `rules.json` — no Python changes
needed:

```json
{
  "name": "R7",
  "conditions": ["cough", "sore_throat"],
  "conclusion": "suspect_bronchitis",
  "certainty_factor": 0.6,
  "explanation": "Persistent cough with sore throat can indicate bronchitis."
}
```

To use an entirely different domain (e.g. loan eligibility, device
troubleshooting), swap in a different rules file: `--rules my_domain.json`.


