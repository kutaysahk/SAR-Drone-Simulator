"""
logic/knowledge_base.py
========================
Propositional Logic engine for Project Aegis.

Improvements over original
---------------------------
1. Rule deduplication  – rules are stored as a set of frozen tuples so the
   same implication is never added twice.  The original appended a new tuple
   on every call to add_rule(), causing the rule list to grow by ~9 entries
   per step and making each infer() pass O(steps^2) by the end of a long run.

2. Hazard / Blocked inference  – if a cell is known Blocked or Heat, the KB
   explicitly refuses to mark it Safe.  Prevents the logic gate in move_to()
   from being bypassed when the camera sees a hazard the KB hasn't registered.

3. Contradiction guard  – if a cell is already Safe it can't become Blocked,
   and vice-versa.  Avoids inconsistent KB state.

4. Efficient incremental forward-chaining  – infer() uses a premise→rules
   reverse index and an agenda queue.  Only rules reachable from newly added
   facts are evaluated, reducing per-step cost from O(|rules|) to
   O(|triggered rules|) in the common case.
"""

from __future__ import annotations


class KnowledgeBase:
    def __init__(self):
        self.facts: set[str] = set()

        # Deduplication: store rules as (frozenset(premises), conclusion)
        self._rule_set: set[tuple] = set()
        self._rules: list[tuple[list[str], str]] = []

        # Reverse index: premise_fact -> [rule_indices that have it as premise]
        self._premise_index: dict[str, list[int]] = {}

        # Facts added since last infer() — feeds the agenda
        self._new_facts: set[str] = set()

        # Full proof trace
        self.inference_trace: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(self, fact: str) -> bool:
        """Add a fact. Returns True if genuinely new. Enforces contradiction guard."""
        if fact in self.facts:
            return False

        # Contradiction guard -------------------------------------------
        if fact.startswith("Safe_"):
            coords = fact[5:]
            if f"Blocked_{coords}" in self.facts or f"Heat_{coords}" in self.facts:
                return False   # refuse to mark a known-dangerous cell as Safe

        if fact.startswith(("Blocked_", "Heat_")):
            # coords follow the first underscore after the prefix
            coords = "_".join(fact.split("_")[1:])
            if f"Safe_{coords}" in self.facts:
                return False   # refuse to mark a known-safe cell as dangerous
        # ---------------------------------------------------------------

        self.facts.add(fact)
        self._new_facts.add(fact)
        return True

    def add_rule(self, premises: list[str], conclusion: str) -> bool:
        """
        Add a Modus-Ponens implication rule.
        Duplicate rules (same premises + conclusion) are silently ignored.
        Returns True if the rule was new.
        """
        key = (frozenset(premises), conclusion)
        if key in self._rule_set:
            return False

        self._rule_set.add(key)
        idx = len(self._rules)
        self._rules.append((list(premises), conclusion))

        # Incrementally update the reverse index
        for p in premises:
            self._premise_index.setdefault(p, []).append(idx)

        return True

    def infer(self) -> int:
        """
        Incremental forward chaining with Modus Ponens.

        Uses an agenda seeded with facts added since the last call.
        Only propagates through rules reachable from those new facts,
        so the cost per step stays proportional to what actually changed.

        Returns the number of new facts deduced.
        """
        total_new = 0
        queue = list(self._new_facts)
        self._new_facts.clear()
        visited_rules: set[int] = set()

        while queue:
            trigger = queue.pop()
            for rule_idx in self._premise_index.get(trigger, []):
                if rule_idx in visited_rules:
                    continue
                visited_rules.add(rule_idx)

                premises, conclusion = self._rules[rule_idx]
                if conclusion in self.facts:
                    continue
                if all(p in self.facts for p in premises):
                    if self.add_fact(conclusion):
                        total_new += 1
                        self.inference_trace.append(
                            f"Modus Ponens: {premises} → {conclusion}"
                        )
                        queue.append(conclusion)

        return total_new

    def ask(self, query: str) -> bool:
        """Return True if the query is a proven fact."""
        return query in self.facts

    def explain_recent_inferences(self, limit: int = 5) -> list[str]:
        """Return the last `limit` inference steps for the proof trace."""
        return self.inference_trace[-limit:]
