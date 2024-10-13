from typing import Collection

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel

from vrp.common import Location
from vrp.problem_instance import ProblemInstance


class TSPSolution(BaseModel, frozen=True):
    locations_in_order: tuple[Location, ...]
    distance: float

    def __str__(self) -> str:
        path_str = "Path: " + " -> ".join(str(loc.id) for loc in self.locations_in_order)
        cost_str = f"Cost: {self.distance}"
        return f"{path_str}\n{cost_str}"


class TSPSolver:
    def __init__(
        self,
        problem_instance: ProblemInstance,
        distance_scaling_factor: float = 100,
    ):
        self.problem_instance = problem_instance
        # scaling factor used to convert distances to integer with limited loss
        self.distance_scaling_factor = distance_scaling_factor

    def solve(self, locations: Collection[Location]) -> TSPSolution:
        locations_with_depot = [self.problem_instance.depot, *locations]
        idx_to_location_map = {idx: loc for loc, idx in zip(locations_with_depot, range(len(locations_with_depot)))}
        depot_idx = 0

        # manages map between variable index and location idx
        idx_manager = pywrapcp.RoutingIndexManager(
            len(locations_with_depot),
            1,
            depot_idx,
        )
        routing_model = pywrapcp.RoutingModel(idx_manager)

        def _distance_callback(from_var_idx: int, to_var_idx: int) -> int:
            # map from variable to loc index
            from_idx = idx_manager.IndexToNode(from_var_idx)
            to_idx = idx_manager.IndexToNode(to_var_idx)
            # map from loc index to loc object
            from_loc = idx_to_location_map[from_idx]
            to_loc = idx_to_location_map[to_idx]
            distance = self.problem_instance.get_distance(from_loc, to_loc)
            return int(distance * self.distance_scaling_factor)

        # add distance information
        transit_callback_index = routing_model.RegisterTransitCallback(_distance_callback)
        routing_model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # define heuristic for initial solution
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        # solve model
        solution = routing_model.SolveWithParameters(search_parameters)
        assert solution is not None, "No solution found"

        # create solution object
        route: list[Location] = []
        index = routing_model.Start(0)
        route.append(idx_to_location_map[idx_manager.IndexToNode(index)])
        route_distance = 0
        while not routing_model.IsEnd(index):
            previous_index = index
            index = solution.Value(routing_model.NextVar(index))
            route.append(idx_to_location_map[idx_manager.IndexToNode(index)])
            route_distance += routing_model.GetArcCostForVehicle(previous_index, index, 0) / float(self.distance_scaling_factor)

        non_depot_locations_in_order = [loc for loc in route if not loc.is_depot]
        return TSPSolution(locations_in_order=tuple(non_depot_locations_in_order), distance=route_distance)


if __name__ == "__main__":
    problem_instance = ProblemInstance.from_csv(
        "instances/example.csv",
        num_vehicles=3,
        vehciel_capacity=15,
    )
    solver = TSPSolver(problem_instance=problem_instance)
    locations = problem_instance.locations
    solution = solver.solve(locations=locations)
    print(solution)
