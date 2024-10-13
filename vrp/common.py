from typing import Optional

from pydantic import BaseModel


class Location(BaseModel, frozen=True):
    loc_x: float
    loc_y: float
    id: str
    demand: Optional[float]
    is_depot: bool = False


class Depot(Location, frozen=True):
    # depot has no delivery requirement
    demand: Optional[float] = None
    is_depot: bool = True
