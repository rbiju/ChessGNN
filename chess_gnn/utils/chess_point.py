from typing import NamedTuple, Tuple, List
from chess import Move


class ChessPoint(NamedTuple):
    x: int
    y: int

    @classmethod
    def from_1d(cls, idx: int) -> "ChessPoint":
        return cls(idx % 8, idx // 8)

    def to_1d(self) -> int:
        return self.x + self.y * 8

    def __add__(self, other: "ChessPoint") -> "ChessPoint":
        return ChessPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "ChessPoint") -> "ChessPoint":
        return ChessPoint(self.x - other.x, self.y - other.y)

    def __eq__(self, other: "ChessPoint") -> bool:
        return self.x == other.x and self.y == other.y

    def up(self):
        return ChessPoint(self.x, self.y + 1)

    def down(self):
        return ChessPoint(self.x, self.y - 1)

    def left(self):
        return ChessPoint(self.x - 1, self.y)

    def right(self):
        return ChessPoint(self.x + 1, self.y)

    @staticmethod
    def _clip_coord(coord: int) -> int:
        return min(max(coord, 0), 8)

    def clip(self):
        return ChessPoint(self._clip_coord(self.x),
                          self._clip_coord(self.y))

    def range(self, other: "ChessPoint") -> List["ChessPoint"]:
        out = []
        for x in range(self.x, other.x + 1):
            for y in range(self.y, other.y + 1):
                out.append(ChessPoint(x, y))

        return out

    def adjacent(self):
        bottom_left = self.down().left().clip()
        top_right = self.up().right().clip()
        return bottom_left.range(top_right)

    def edge(self, other: "ChessPoint") -> "ChessEdge":
        return ChessEdge(self, other)


class ChessEdge(NamedTuple):
    from_square: ChessPoint
    to_square: ChessPoint

    @classmethod
    def from_move(cls, move: Move) -> "ChessEdge":
        return cls(ChessPoint.from_1d(move.from_square), ChessPoint.from_1d(move.to_square))

    def to_1d(self) -> Tuple[int, int]:
        return self.from_square.to_1d(), self.to_square.to_1d()
