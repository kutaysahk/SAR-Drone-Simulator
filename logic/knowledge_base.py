class KnowledgeBase:
    def __init__(self):
        # Stores absolute truths (e.g., "Safe_0_0", "Clear_0_0", "Heat_1_1")
        self.facts = set() 
        # Stores rules as tuples: (list_of_premises, conclusion)
        self.rules = []    
        self.inference_trace = []

    def add_fact(self, fact: str):
        """Adds a fact to the KB if it doesn't already exist."""
        if fact not in self.facts:
            self.facts.add(fact)
            return True
        return False

    def add_rule(self, premises: list, conclusion: str):
        """
        Adds a rule for Modus Ponens inference.
        Example: premises=["Clear_0_0"], conclusion="Safe_1_0"
        """
        self.rules.append((premises, conclusion))

    def infer(self) -> int:
        """
        Applies Modus Ponens: If all premises of a rule are known facts,
        the conclusion becomes a new known fact.
        Returns the number of new facts deduced in this cycle.
        """
        new_facts_count = 0
        inferred_something = True

        # Keep inferring until no new facts can be deduced (Forward Chaining)
        while inferred_something:
            inferred_something = False
            for premises, conclusion in self.rules:
                # Check if all premises are in our known facts
                if all(premise in self.facts for premise in premises):
                    # If we don't already know the conclusion, add it!
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        self.inference_trace.append(
                            f"Modus Ponens: {premises} -> {conclusion}"
                        )
                        inferred_something = True
                        new_facts_count += 1
                        
        return new_facts_count

    def ask(self, query: str) -> bool:
        """Queries the KB to see if a fact is proven true."""
        return query in self.facts

    def explain_recent_inferences(self, limit: int = 5) -> list:
        """Returns a compact proof trace for demonstration and debugging."""
        return self.inference_trace[-limit:]
