# Project Aegis

Project Aegis is a small search-and-rescue drone simulator for a disaster zone. The
agent explores a hidden grid, avoids rubble and hazards, and prioritizes cells that
maximize rescue value.

## Simulation

The app starts with a single entry screen. Press Start Simulation to generate a
disaster grid with rubble, fire, and hidden survivors.

## 1. Logic

The agent uses propositional logic in `logic/knowledge_base.py`.

- Facts are symbols such as `Clear_0_0`, `Safe_1_0`, and `Heat_4_2`.
- Rules are implication clauses such as `Clear_0_0 -> Safe_1_0`.
- Inference uses forward chaining with Modus Ponens:
  - If `Clear_0_0` is true.
  - And the rule `Clear_0_0 -> Safe_1_0` exists.
  - Then the agent infers `Safe_1_0`.

The movement gate in `state/drone.py` asks the knowledge base whether a target cell
is proven safe before the drone can move there.

## Pathfinding and Exploration

The drone uses nearest-neighbor frontier exploration in `state/drone.py`. After
every scan it replans from its current position, so newly discovered nearby gaps are
checked before the drone continues toward a farther target.

A frontier is a known-safe travel cell whose full 3x3 sensor footprint contains at
least one unchecked cell. Breadth-first search (BFS) finds the shortest reachable
path to the nearest frontier through known safe cells.

This makes the drone systematically check reachable unexplored places in shortest
path order instead of choosing moves from only its recent history or following stale
plans.

## 2. The Math of AI

The decision model in `state/drone.py` is an expected utility calculation using
linear algebra and probability.

Each candidate move is represented as a feature vector:

```text
x = [is_unexplored, P(survivor), normalized_distance]
```

The optimizer learns a weight vector:

```text
w = [w_explore, w_probability, w_distance]
```

The drone scores a move with a dot product:

```text
E[U] = [w_explore, w_probability, -w_distance] dot x
```

The `P(survivor)` term comes from the environment's survivor probability
distribution. The `normalized_distance` term is calculated from a Euclidean norm.

## 3. Optimization

The selected optimization method is a Genetic Algorithm in
`optimization/genetic_algo.py`.

- Chromosomes encode the utility weights.
- Fitness evaluates weights on representative rescue scenarios.
- Selection keeps the best half of the population.
- Crossover mixes parent chromosomes.
- Mutation adds bounded random variation.

Distance is the first priority for nearest-neighbor coverage. When several frontier
paths have the same shortest distance, the optimized expected utility model chooses
the frontier that reveals the most useful unchecked territory.

## Rescue Beacons

When the drone senses a survivor, it prints a help call and stores that location as
a rescue beacon. The visualizer draws a blinking green marker on every discovered
survivor cell.

## Running

Install dependencies:

```powershell
pip install -r requirements.txt.txt
```

Start the simulator:

```powershell
py -3.10 main.py
```

Hold Space while the window is focused to reveal the ground-truth disaster map.
Press Esc during a run to return to the start screen.
