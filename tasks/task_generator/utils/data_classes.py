from dataclasses import dataclass

@dataclass
class Block:
	identifier: int
	type: str
	material: str
	x: float
	y: float
	rotation: float
	scale_x: float = 1.0
	scale_y: float = 1.0


@dataclass
class Pig:
	identifier: int
	type: str
	x: float
	y: float
	rotation: float


@dataclass
class Tnt:
	identifier: int
	x: float
	y: float
	rotation: float


@dataclass
class Bird:
	type: str

