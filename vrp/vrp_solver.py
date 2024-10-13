from collections import defaultdict
from itertools import chain, combinations

import numpy as np
import pandas as pd
import plotly.express as px
from ortools.linear_solver import pywraplp
from pydantic import BaseModel, NonNegativeFloat, field_validator

from vrp.common import Location
from vrp.problem_instance import ProblemInstance
from vrp.tsp_solver import TSPSolver


class Route(BaseModel, frozen=True):
    locations_in_order: tuple[Location, ...]
    distance: NonNegativeFloat
    capacity_used: NonNegativeFloat
    total_capacity: NonNegativeFloat
    generation_model_name: str

    @field_validator("locations_in_order")
    def depot_is_first_location(cls, locations_in_order: tuple[Location, ...]) -> tuple[Location, ...]:
        assert locations_in_order, "route must have at least one location"
        assert locations_in_order[0].is_depot, "first location must be depot"
        return locations_in_order

    def __str__(self) -> str:
        path_str = " -> ".join(str(loc.id) for loc in self.locations_in_order)
        return f"{path_str}"

    @classmethod
    def from_locations_in_order(
        cls,
        locations_in_order: list[Location],
        problem_instance: ProblemInstance,
        model_name: str,
    ) -> "Route":
        assert locations_in_order[0].is_depot, "first location must be depot"
        distance = 0
        capacity_used = 0
        total_capacity = problem_instance.vehicle_capacity
        for i, loc in enumerate(locations_in_order):
            if i == 0:
                continue
            distance += problem_instance.get_distance(locations_in_order[i - 1], loc)
            assert loc.demand is not None
            capacity_used += loc.demand
        # add distance back to depot
        distance += problem_instance.get_distance(locations_in_order[-1], locations_in_order[0])
        assert cls.is_feasible_route(locations_in_order, problem_instance), "route is infeasible"
        return cls(
            locations_in_order=tuple(locations_in_order),
            distance=distance,
            capacity_used=capacity_used,
            total_capacity=total_capacity,
            generation_model_name=model_name,
        )

    @staticmethod
    def is_feasible_route(
        locations: list[Location],
        problem_instance: ProblemInstance,
    ) -> bool:
        capacity_used = sum([loc.demand for loc in locations if loc.demand is not None])
        return capacity_used <= problem_instance.vehicle_capacity


class VRPSolution(BaseModel, frozen=True):
    routes: tuple[Route, ...]
    considered_routes: set[Route]
    distance: float

    def __str__(self) -> str:
        routes_str = "\n\n".join(f"Path {i}: {str(route)}" for i, route in enumerate(self.routes))
        cost_str = f"Total Cost: {self.distance}"
        return f"\n{routes_str}\n\n{cost_str}\n"

    def visualize(self, size: float = 10.0, show: bool = False):
        data = []
        for idx, r in enumerate(self.routes):
            x_y_route_id_loc_id = [(loc.loc_x, loc.loc_y, idx, loc.id, r.generation_model_name) for loc in r.locations_in_order]
            # add depot to end of route
            x_y_route_id_loc_id.append((r.locations_in_order[0].loc_x, r.locations_in_order[0].loc_y, idx, r.locations_in_order[0].id, ""))
            data.extend(x_y_route_id_loc_id)

        frame = pd.DataFrame(data, columns=["x coordinate", "y coordinate", "Route ID", "Location ID", "Generation Model"])

        fig = px.line(
            frame,
            x="x coordinate",
            y="y coordinate",
            color="Route ID",
            title="Vehicle Routing Problem",
            hover_name="Route ID",
            hover_data=["x coordinate", "y coordinate", "Location ID", "Generation Model"],
            markers=True,
        )
        fig.update_layout(
            height=size * 100,
            width=size * 100,
        )
        if show:
            fig.show()
        return fig


class VRPInitialRouteHeuristic:
    def __init__(
        self,
        tsp_correct_paths: bool = True,
    ) -> None:
        self.tsp_correct_paths = tsp_correct_paths

    def get_routes(self, problem_instance: ProblemInstance) -> set[Route]:
        location_lists = self._get_routes(problem_instance)
        if self.tsp_correct_paths:
            tsp_solver = TSPSolver(problem_instance)
            location_lists = [tsp_solver.solve(s).locations_in_order for s in location_lists]
        routes = set(
            Route.from_locations_in_order([problem_instance.depot, *s], problem_instance, model_name=self._get_model_name())
            for s in location_lists
            if Route.is_feasible_route([problem_instance.depot, *s], problem_instance)
        )
        return routes

    def _get_routes(self, problem_instance: ProblemInstance) -> set[tuple[Location, ...]]:
        raise NotImplementedError

    def _get_model_name(self) -> str:
        return self.__class__.__name__


class ExhaustiveHeuristic(VRPInitialRouteHeuristic):
    def _get_routes(self, problem_instance: ProblemInstance) -> set[tuple[Location, ...]]:
        if not self.tsp_correct_paths:
            tsp_ordering = TSPSolver(problem_instance).solve(problem_instance.locations)
            tsp_ranking = {loc: i for i, loc in enumerate(tsp_ordering.locations_in_order)}
        else:
            tsp_ranking = {loc: i for i, loc in enumerate(problem_instance.locations)}

        all_locations = [loc for loc in problem_instance.locations if not loc.is_depot]
        all_location_subsets = chain.from_iterable(combinations(all_locations, r) for r in range(1, len(all_locations) + 1))
        all_location_subsets = [tuple(sorted(s, key=lambda x: tsp_ranking[x])) for s in all_location_subsets]
        return set(all_location_subsets)


class NearestNeighborHeuristic(VRPInitialRouteHeuristic):
    def _get_routes(self, problem_instance: ProblemInstance) -> set[tuple[Location, ...]]:
        if not self.tsp_correct_paths:
            tsp_ordering = TSPSolver(problem_instance).solve(problem_instance.locations)
            tsp_ranking = {loc: i for i, loc in enumerate(tsp_ordering.locations_in_order)}
        else:
            tsp_ranking = {loc: i for i, loc in enumerate(problem_instance.locations)}

        location_lists = set()
        for location in [problem_instance.depot, *problem_instance.locations]:
            # order all locations by distance from current location
            locations_by_distance = sorted(
                [loc for loc in problem_instance.locations if loc != location and not loc.is_depot],
                key=lambda x: problem_instance.get_distance(location, x),
            )
            location_lists_by_distance = [locations_by_distance[:i] for i in range(1, len(locations_by_distance) + 1)]
            location_lists_by_distance = [tuple(sorted(s, key=lambda x: tsp_ranking[x])) for s in location_lists_by_distance]
            location_lists.update(location_lists_by_distance)
        return location_lists


class SweepAngleHeuristic(VRPInitialRouteHeuristic):
    def __init__(
        self,
        tsp_correct_paths: bool = True,
        min_capacity_ratio: float = 0.5,
    ) -> None:
        super().__init__(tsp_correct_paths=tsp_correct_paths)
        self.min_capacity_ratio = min_capacity_ratio

    @staticmethod
    def _get_angles(loc_x_y: list[tuple[float, float]]) -> list[float]:
        # returns angle of locations vs north pole
        # values in [0, 2pi)]
        vectors = np.array(loc_x_y).T
        angles = np.arctan2(vectors[0], vectors[1])
        scaled_angles = angles + 2 * np.pi * np.where(angles < 0, 1, 0)
        return list(scaled_angles)

    def _get_routes(self, problem_instance: ProblemInstance) -> set[tuple[Location, ...]]:
        locations = [loc for loc in problem_instance.locations if not loc.is_depot]
        depot_x_y = problem_instance.depot.loc_x, problem_instance.depot.loc_y
        location_x_y = [(loc.loc_x, loc.loc_y) for loc in problem_instance.locations if not loc.is_depot]
        normalized_loc_x_y = [(x - depot_x_y[0], y - depot_x_y[1]) for x, y in location_x_y]
        # take angle of location vs north pole
        angles = self._get_angles(normalized_loc_x_y)
        ordered_locations = [loc for _, loc in sorted(zip(angles, locations), key=lambda x: x[0])]

        ordered_locations_sweep_list = ordered_locations * 2

        all_sweeps = []
        for i in range(len(ordered_locations)):
            # add all sweeps starting from index
            sweeps = [tuple(ordered_locations_sweep_list[i : i + j]) for j in range(1, len(ordered_locations))]
            capacity_ratios = [sum(loc.demand for loc in s if loc.demand is not None) / problem_instance.vehicle_capacity for s in sweeps]
            capacity_constraint_sweeps = [
                s for i, s in enumerate(sweeps) if capacity_ratios[i] >= self.min_capacity_ratio and capacity_ratios[i] <= 1
            ]
            all_sweeps.extend(capacity_constraint_sweeps)
        return set(all_sweeps)


class VRPSolver:
    def __init__(
        self,
        problem_instance: ProblemInstance,
        initial_route_heuristics: list[VRPInitialRouteHeuristic],
    ):
        self.problem_instance = problem_instance
        assert initial_route_heuristics, "must provide at least one route heuristic"
        self.initial_route_heuristics = initial_route_heuristics

    def _find_best_covering_routes(self, routes: tuple[Route, ...]) -> tuple[Route, ...]:
        location_to_covering_routes = defaultdict(list)
        for r in routes:
            for location in r.locations_in_order:
                if location.is_depot:
                    continue
                location_to_covering_routes[location].append(r)

        solver = pywraplp.Solver.CreateSolver("SCIP")
        assert solver is not None, "Solver not created"

        # create route variables
        route_vars = {}
        for r in routes:
            route_vars[r] = solver.IntVar(0, 1, f"route_var({r})")

        # each location must be covered by a route
        for loc in self.problem_instance.locations:
            if loc.is_depot:
                continue
            constraint_expr = [route_vars[r] for r in location_to_covering_routes[loc]]
            solver.Add(sum(constraint_expr) >= 1)

        # number of routes at most number of vehicles
        solver.Add(solver.Sum(list(route_vars.values())) <= self.problem_instance.num_vehicles)

        # define objective
        obj_expr = [route_vars[r] * r.distance for r in routes]
        solver.Minimize(solver.Sum(obj_expr))

        status = solver.Solve()
        assert status == pywraplp.Solver.OPTIMAL, "No optimal solution found"

        best_routes = tuple(r for r in routes if route_vars[r].solution_value() > 0.5)
        return best_routes

    def _shorten_routes(self, routes: tuple[Route, ...]) -> tuple[Route, ...]:
        short_routes = []
        for r in routes:
            locations = [loc for loc in r.locations_in_order if not loc.is_depot]
            tsp_solution = TSPSolver(self.problem_instance).solve(locations)
            short_routes.append(
                Route.from_locations_in_order(
                    [self.problem_instance.depot, *tsp_solution.locations_in_order],
                    self.problem_instance,
                    model_name=r.generation_model_name,
                )
            )
        return tuple(short_routes)

    def solve(self) -> VRPSolution:
        routes = set()
        for heuristic in self.initial_route_heuristics:
            routes |= heuristic.get_routes(self.problem_instance)
        best_routes = self._find_best_covering_routes(tuple(routes))
        short_routes = self._shorten_routes(best_routes)
        total_distance = sum(r.distance for r in short_routes)
        return VRPSolution(routes=short_routes, considered_routes=routes, distance=total_distance)


if __name__ == "__main__":
    problem_instance = ProblemInstance.from_csv(
        "instances/example.csv",
        num_vehicles=3,
        vehciel_capacity=15,
    )
    # problem_instance = ProblemInstance.from_random(num_locations=15, num_vehicles=4)
    solver = VRPSolver(
        problem_instance,
        [SweepAngleHeuristic(), NearestNeighborHeuristic()],
    )
    sol = solver.solve()
    print(sol)
