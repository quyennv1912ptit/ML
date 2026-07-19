from typing import Any

class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'depth', 'value', 'n_samples', 'impurity')

    def __init__(self, depth: int):
            self.feature: int | None = None
            self.threshold: float | None= None
            self.left: Node | None = None
            self.right: Node | None = None
            self.depth: int = depth
            self.value: Any = None
            self.n_samples: int | None = None
            self.impurity: float | None = None

    def is_leaf(self):
        return self.value is not None
