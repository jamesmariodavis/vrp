from enum import Enum
from itertools import product
from typing import Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel

from vrp.common import Depot, Location


class ColumnMapping(Enum):
    LOCATION = "Location"
    X = "Coodinate: X"
    Y = "Coordinate: Y"
    DEMAND = "Delivery Requirement (unit)"


class ProblemInstance(BaseModel):
    depot: Depot
    locations: list[Location]
    distances: dict[tuple[Location, Location], float]
    num_vehicles: int
    vehicle_capacity: float

    def get_distance(
        self,
        from_loc: Location,
        to_loc: Location,
    ) -> float:
        return self.distances[(from_loc, to_loc)]

    @staticmethod
    def _get_2_norm(from_loc: Location, to_loc: Location) -> float:
        distance = ((from_loc.loc_x - to_loc.loc_x) ** 2 + (from_loc.loc_y - to_loc.loc_y) ** 2) ** 0.5
        assert distance >= 0
        return distance

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        num_vehicles: int,
        vehciel_capacity: float,
        col_mapping: Type[ColumnMapping] = ColumnMapping,
    ) -> "ProblemInstance":

        data = pd.read_csv(file_path)

        data[col_mapping.DEMAND.value] = data[col_mapping.DEMAND.value].astype(float)
        data[col_mapping.LOCATION.value] = data[col_mapping.LOCATION.value].astype(str)
        for col in [col_mapping.X, col_mapping.Y]:
            data[col.value] = data[col.value].astype(float)
        depot_rows = data[data[col_mapping.LOCATION.value] == "Depot"]
        location_rows = data[data[col_mapping.LOCATION.value] != "Depot"]

        assert len(depot_rows) == 1
        assert len(location_rows) > 0
        assert len(location_rows[col_mapping.LOCATION.value].unique()) == len(location_rows)

        depot = Depot(
            loc_x=depot_rows.iloc[0][col_mapping.X.value],
            loc_y=depot_rows.iloc[0][col_mapping.Y.value],
            id=depot_rows.iloc[0][col_mapping.LOCATION.value],
        )
        locations = [
            Location(
                id=row[col_mapping.LOCATION.value],
                loc_x=row[col_mapping.X.value],
                loc_y=row[col_mapping.Y.value],
                demand=row[col_mapping.DEMAND.value],
            )
            for _, row in location_rows.iterrows()
        ]
        distances = {(from_loc, to_loc): cls._get_2_norm(from_loc, to_loc) for from_loc, to_loc in product([depot, *locations], repeat=2)}
        assert len(distances) == (len(locations) + 1) ** 2
        return cls(
            depot=depot,
            locations=locations,
            distances=distances,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehciel_capacity,
        )

    @classmethod
    def from_google_vrp_instance(cls) -> "ProblemInstance":
        # https://developers.google.com/optimization/routing/vrp
        # https://developers.google.com/optimization/routing/cvrp
        distance_matrix = [
            [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
            [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
            [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
            [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
            [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
            [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
            [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
            [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
            [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
            [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
            [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
            [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
            [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
            [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
            [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
            [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
            [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
        ]
        locations = [
            (456, 320),
            (228, 0),
            (912, 0),
            (0, 80),
            (114, 80),
            (570, 160),
            (798, 160),
            (342, 240),
            (684, 240),
            (570, 400),
            (912, 400),
            (114, 480),
            (228, 480),
            (342, 560),
            (684, 560),
            (0, 640),
            (798, 640),
        ]
        demands = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
        depot = Depot(loc_x=locations[0][0], loc_y=locations[0][1], id="0")
        locations = [Location(loc_x=x, loc_y=y, id=str(i), demand=demands[i - 1]) for i, (x, y) in enumerate(locations[1:], 1)]
        depot_and_locations = [depot, *locations]
        distances = {
            (depot_and_locations[i], depot_and_locations[j]): float(distance_matrix[i][j])
            for i, j in product(range(len(depot_and_locations)), repeat=2)
        }
        return cls(
            depot=depot,
            locations=locations,
            distances=distances,
            num_vehicles=4,
            vehicle_capacity=15,
        )

    @classmethod
    def from_random(
        cls,
        num_locations: int,
        num_vehicles: Optional[int] = None,
        centered_depot: bool = True,
        capacity_overage: float = 1.2,
    ) -> "ProblemInstance":
        if num_vehicles is None:
            num_vehicles = max(num_locations // 4 + 1, 3)
        rng = np.random.default_rng(seed=0)
        if centered_depot:
            depot = Depot(loc_x=0, loc_y=0, id="0")
        else:
            depot = Depot(loc_x=-100, loc_y=-100, id="0")
        locations = [
            Location(
                loc_x=rng.uniform(-100, 100),
                loc_y=rng.uniform(-100, 100),
                id=str(i),
                demand=rng.integers(1, 10),
            )
            for i in range(1, num_locations + 1)
        ]
        total_demand = sum(loc.demand for loc in locations if loc.demand is not None)
        vehicle_capacity = total_demand * capacity_overage / num_vehicles
        distances = {(from_loc, to_loc): cls._get_2_norm(from_loc, to_loc) for from_loc, to_loc in product([depot, *locations], repeat=2)}
        return cls(
            depot=depot,
            locations=locations,
            distances=distances,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
        )


if __name__ == "__main__":
    problem_instance = ProblemInstance.from_csv(
        "instances/example.csv",
        col_mapping=ColumnMapping,
        num_vehicles=3,
        vehciel_capacity=15,
    )
