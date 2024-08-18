from dataclasses import dataclass
from typing import Optional


@dataclass
class Material:
    E: float
    G: float
    density: float

    name: Optional[str] = None