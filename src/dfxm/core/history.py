"""Immutable ordered sequence of :class:`Operation` — the processing recipe.

Being immutable (a tuple under the hood), every edit returns a new History,
which is exactly what the immutable :class:`~dfxm.core.dataset.DFXMDataset`
needs. Serializes to / from a plain list for SQLite session storage and Replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ops import Operation


@dataclass(frozen=True)
class History:
    ops: tuple = ()

    def add(self, op: Operation) -> History:
        return History((*self.ops, op))

    def pop(self) -> History:
        """Drop the last op (Undo)."""
        return History(self.ops[:-1]) if self.ops else self

    def replace_all(self, ops) -> History:
        return History(tuple(ops))

    def __iter__(self):
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __getitem__(self, i):
        return self.ops[i]

    def to_list(self) -> list[dict]:
        return [op.to_dict() for op in self.ops]

    @classmethod
    def from_list(cls, items) -> History:
        return cls(tuple(Operation.from_dict(d) for d in (items or [])))
