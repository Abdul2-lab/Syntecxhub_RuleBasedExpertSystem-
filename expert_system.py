"""
Rule-Based Expert System (v2)
------------------------------
A forward-chaining inference engine with:

  - Data-driven rules  : rules/facts live in rules.json, not hardcoded in
                          Python, so the knowledge base can be edited or
                          swapped without touching the engine code.
  - Certainty factors  : each rule carries a confidence (0-1), and derived
                          facts accumulate certainty using a MYCIN-style
                          combination formula, instead of being purely
                          boolean true/false.
  - Multi-step chaining: a derived fact can satisfy the condition of another
                          rule, so conclusions build on each other.
  - Structured logging : uses Python's `logging` module (console + optional
                          log file) instead of scattered print statements.
  - CLI                : run interactively, from a fixed fact list, or in
                          batch/non-interactive mode via command-line flags
                          -- useful for automation / grading scripts.

Usage
-----
    python3 expert_system.py                       # interactive mode
    python3 expert_system.py --facts fever,cough    # batch mode
    python3 expert_system.py --demo                 # built-in sample run
    python3 expert_system.py --log-file run.log     # also write log to file
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("expert_system")


def configure_logging(log_file: str | None = None, verbose: bool = True) -> None:
    """
    Configure console (and optionally file) logging for the engine.

    verbose=True  -> console shows every FACT/RULE step (DEBUG+).
    verbose=False -> console only shows warnings/errors (i.e. --quiet mode),
                     while the optional log file still records everything.
    """
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter it
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """A single IF-THEN production rule with an associated certainty factor."""

    name: str
    conditions: Set[str]
    conclusion: str
    certainty_factor: float = 1.0     # how reliable this rule itself is (0-1)
    explanation: str = ""

    def is_satisfied_by(self, known_facts: Set[str]) -> bool:
        return self.conditions.issubset(known_facts)

    @staticmethod
    def from_dict(data: dict) -> "Rule":
        return Rule(
            name=data["name"],
            conditions=set(data["conditions"]),
            conclusion=data["conclusion"],
            certainty_factor=float(data.get("certainty_factor", 1.0)),
            explanation=data.get("explanation", ""),
        )


@dataclass
class KnowledgeBase:
    """Holds rules plus the growing set of facts with their certainty (0-1)."""

    rules: List[Rule]
    facts: Dict[str, float] = field(default_factory=dict)

    def add_fact(self, fact: str, certainty: float = 1.0) -> None:
        """
        Add or update a fact. If the fact already exists, combine the two
        certainty values using the classic MYCIN combination rule:

            CF_combined = CF_old + CF_new * (1 - CF_old)

        This lets multiple rules independently support the same conclusion
        and increase confidence in it, rather than one simply overwriting
        the other.
        """
        certainty = max(0.0, min(1.0, certainty))
        if fact in self.facts:
            old = self.facts[fact]
            combined = old + certainty * (1 - old)
            self.facts[fact] = combined
            logger.debug(
                f"[FACT UPDATED] '{fact}': {old:.2f} -> {combined:.2f}"
            )
        else:
            self.facts[fact] = certainty
            logger.debug(f"[FACT ADDED] '{fact}' (certainty={certainty:.2f})")

    def fact_names(self) -> Set[str]:
        return set(self.facts.keys())

    def certainty_of(self, fact: str) -> float:
        return self.facts.get(fact, 0.0)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """Forward-chaining engine that operates over a KnowledgeBase."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.fired_rules: List[str] = []

    def _rule_conclusion_certainty(self, rule: Rule) -> float:
        """
        Certainty of a fired rule's conclusion = rule's own certainty_factor
        multiplied by the weakest (minimum-certainty) supporting condition
        -- a fuzzy-AND, since a chain is only as strong as its weakest link.
        """
        min_condition_certainty = min(
            self.kb.certainty_of(c) for c in rule.conditions
        )
        return rule.certainty_factor * min_condition_certainty

    def run(self, max_passes: int = 50) -> List[str]:
        """
        Repeatedly scan all rules and fire any whose conditions are met,
        deriving new facts (with certainty). Stops when a full pass adds no
        new facts, or after `max_passes` safety-limit iterations.
        """
        changed = True
        passes = 0

        while changed and passes < max_passes:
            changed = False
            passes += 1
            for rule in self.kb.rules:
                if rule.is_satisfied_by(self.kb.fact_names()):
                    new_certainty = self._rule_conclusion_certainty(rule)
                    existing = self.kb.certainty_of(rule.conclusion)
                    # Only count as "firing" if it meaningfully changes things
                    if rule.conclusion not in self.kb.facts or new_certainty > existing:
                        self.kb.add_fact(rule.conclusion, new_certainty)
                        self.fired_rules.append(rule.name)
                        logger.info(
                            f"[RULE FIRED] {rule.name}: "
                            f"IF {sorted(rule.conditions)} -> THEN '{rule.conclusion}' "
                            f"(certainty={self.kb.certainty_of(rule.conclusion):.2f})"
                            + (f"  — {rule.explanation}" if rule.explanation else "")
                        )
                        changed = True

        if passes >= max_passes:
            logger.warning("Max inference passes reached — stopping to avoid an infinite loop.")

        return self.fired_rules


# ---------------------------------------------------------------------------
# Loading the knowledge base from JSON
# ---------------------------------------------------------------------------
def load_knowledge_base(rules_path: Path) -> tuple[KnowledgeBase, List[str]]:
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = [Rule.from_dict(r) for r in data["rules"]]
    symptoms = data.get("symptoms", [])
    return KnowledgeBase(rules=rules), symptoms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def ask_symptoms_interactively(symptoms: List[str]) -> Dict[str, float]:
    print("Answer y/n for each symptom (optionally add a confidence 0-1):\n")
    given: Dict[str, float] = {}
    for symptom in symptoms:
        raw = input(f"  Do you have '{symptom}'? (y/n[,confidence]): ").strip().lower()
        if not raw:
            continue
        parts = raw.split(",")
        answer = parts[0].strip()
        if answer == "y":
            certainty = float(parts[1]) if len(parts) > 1 else 1.0
            given[symptom] = certainty
    return given


def print_conclusions(kb: KnowledgeBase, given_facts: Set[str]) -> None:
    derived = {f: c for f, c in kb.facts.items() if f not in given_facts}
    print("\n--- Conclusions (fact: certainty) ---")
    if derived:
        for fact, certainty in sorted(derived.items(), key=lambda x: -x[1]):
            print(f" -> {fact}: {certainty:.2f}")
    else:
        print("No conclusions could be derived from the given facts.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based expert system (forward chaining).")
    parser.add_argument(
        "--rules", type=str, default="rules.json",
        help="Path to the JSON rule base (default: rules.json in this folder).",
    )
    parser.add_argument(
        "--facts", type=str, default=None,
        help="Comma-separated list of starting facts for batch/non-interactive mode, "
             "e.g. --facts fever,cough,body_ache",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with a built-in sample fact set instead of asking for input.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional path to also write the reasoning log to a file.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print final conclusions, not the step-by-step reasoning log.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(log_file=args.log_file, verbose=not args.quiet)

    rules_path = Path(__file__).parent / args.rules
    kb, symptoms = load_knowledge_base(rules_path)

    print("=" * 60)
    print(" Rule-Based Expert System v2 - Symptom Checker (Demo)")
    print("=" * 60)

    if args.demo:
        given = {"fever": 1.0, "cough": 1.0, "body_ache": 1.0}
        print(f"\nRunning demo with facts: {given}")
    elif args.facts:
        given = {name.strip(): 1.0 for name in args.facts.split(",") if name.strip()}
        print(f"\nRunning with provided facts: {given}")
    else:
        given = ask_symptoms_interactively(symptoms)

    for fact, certainty in given.items():
        kb.add_fact(fact, certainty)

    engine = InferenceEngine(kb)
    engine.run()

    print_conclusions(kb, given_facts=set(given.keys()))


if __name__ == "__main__":
    main()
