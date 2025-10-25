# app/cache.py

from collections import OrderedDict
from typing import Any, Tuple

class SimpleLRUCache:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def put(self, key: str, value: Any):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

CACHE = SimpleLRUCache()

def make_key(*parts: Tuple[Any, ...]) -> str:
    return "|".join(map(str, parts))
