import abc
import typing as t

from collections import UserList
from collections.abc import Collection
from heapq import heappush, heappop

from mofa.scoring.graph import AbsoluteScorer, RelativeScorer

T = t.TypeVar("T")


class PriorityBuffer(abc.ABC, UserList[T]):
    """
    An abstract class that extends the Python `list` class.

    Given some maximum number of data items it can hold onto, instances of this
    class will drop low-ranked items based on a user-provided heuristic.
    """

    def __init__(self, heuristic: t.Callable, maxlen: int | None = None):
        """Initializes a Python list with a user-provided list for ranking data items and a maximum length.

        Args:
            heuristic (t.Callable): User-defined heuristic.
            maxlen (int | None): Maximum number of items that can be held in the cache. This parameter
                should not really be set to `None`. In this case, it operates as a standard list.

        Notes:
             Items with the **lowest** heuristic-returned rank/score are dropped from the
             cache (when `maxlen` is exceeded).
        """
        super().__init__()
        self.maxlen = maxlen
        self.heuristic = heuristic

    def append(self, item: T):
        """
        Calls the standard `list.append` function and then prunes if `maxlen` is exceeded.

        Args:
            item (T): Data item to add to the list.
        """
        super().append(item)
        if len(self.data) > self.maxlen:
            self._prune()

    def extend(self, items: Collection[T]):
        """
        Calls the standard `list.extend` function and then prunes if `maxlen` is exceeded.

        Args:
            items (Collection[T]): Data items to be added to the list.
        """
        super().extend(items)
        if len(self.data) > self.maxlen:
            self._prune()

    def _prune(self):
        """
        Prunes (or drops) items from `self.data` until the its length no longer exceeds `maxlen`.

        Notes:
             Items with the **lowest** heuristic-returned rank/score are dropped from the
             cache (when `maxlen` is exceeded).
        """
        ranked_items = self._ranked_items()
        while len(self.data) > self.maxlen:
            rank, item = heappop(ranked_items)
            self.data.remove(item)

    @abc.abstractmethod
    def _ranked_items(self) -> list[T]:
        pass


class AbsolutePriorityBuffer(PriorityBuffer):
    """
    This `PriorityBuffer` applies a user-provided heuristic that scores/ranks each data item
    individually and independently.
    """

    def __init__(self, heuristic: AbsoluteScorer, maxlen: int | None = None):
        """Initializes a Python list with a user-provided list for ranking data items and a maximum length.

        Args:
            heuristic (t.Callable[[T], float]): A function that scores a single data item and returns a single score.
            maxlen (int | None): Maximum number of items that can be held in the cache. Defaults to `None`.
        """
        super().__init__(heuristic, maxlen)

    def _ranked_items(self) -> list[T]:
        ranked_queue = []
        for item in self.data:
            heappush(ranked_queue, (self.heuristic(item), item))
        return ranked_queue


class RelativePriorityBuffer(PriorityBuffer):
    """
    This `PriorityBuffer` applies a user-provided heuristic that scores/ranks the data altogether and, thus,
    can rank the data relative to one another.
    """

    def __init__(
            self,
            heuristic: RelativeScorer,
            maxlen: int | None = None,
    ):
        """Initializes a Python list with a user-provided list for ranking data items and a maximum length.

        Args:
            heuristic (t.Callable[[Collection[T]], Collection[float]]): User-defined heuristic that takes
                an iterable of data items and returns an iterable of floats.
            maxlen (int | None): Maximum number of items that can be held in the cache. Defaults to `None`.
        """
        super().__init__(heuristic, maxlen)

    def _ranked_items(self) -> list[T]:
        ranks = self.heuristic(self.data)
        ranked_queue = []
        for rank, item in zip(ranks, self.data):
            heappush(ranked_queue, (rank, item))
        return ranked_queue


if __name__ == "__main__":
    longest_str_fn = lambda txt: len(txt)  # Heuristic that ranks strings by length.
    q = AbsolutePriorityBuffer(longest_str_fn, maxlen=2)
    q.extend(["very_long_string", "short_str", "str", "medium_str"])
    print(q)
    # >>> ['very_long_string', 'medium_str']
