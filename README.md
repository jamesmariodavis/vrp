To create a virtualenv with all dependencies within the repo run:

```
poetry install
```

## Solution methodology

We implement the following class of heuristic:

1. Find candidate subsets of locations to visit on a single route
2. For each subset of locations solve a TSP problem including the depot to find the most efficient route
3. Solve a capacitated min cost set cover problem (CSC) where elements to be covered are locations and sets are TSP routes from step 2
4. Improve the routes in solution iteratively (not implemented in code)

When we search all subsets of locations in step (1) then we are gauranteed to have an optimal solution although it is extremely computationally burdensome. We leverage this for performance benchmarking on small instances.

## Code structure

### problem_instance.py

Generates problem instances, including parsing csv formats and random generation

### vrp_solver.py

Contains code related to solving the CVRP problem. VRPSolver is the primary class for the solution.

- **ExhaustiveHeuristic** is used within the solver in order to find an optimal solution.
- **NearestNeighborHeuristic** computational efficient heuristic for problem classes with denser demands.
- **SweepAngleHeuristic** computationally efficient heuristic for problem classes with disperese demands.

### tsp_solver.py

Provides a reliable traveling salesman problem (TSP) solver which is leveraged as a subroutine.

### profile.ipynb

Executes code to measure performance relative to optimal and computational efficiency
