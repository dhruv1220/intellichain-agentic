import json
import os
from collections import defaultdict

class MemoryStore:
    def __init__(self, path="logs/user_memory.json"):
        self.path = path
        # Ensure the directory exists before any reads/writes
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r") as f:
                self.store = json.load(f)
        else:
            self.store = defaultdict(dict)
            # Persist initial empty store so future reads succeed
            self._save()

    def get(self, user_id: str) -> dict:
        return self.store.get(user_id, {})

    def set(self, user_id: str, key: str, value):
        if user_id not in self.store:
            self.store[user_id] = {}
        self.store[user_id][key] = value
        self._save()

    def update(self, user_id: str, key: str, value: str):
        if user_id not in self.store:
            self.store[user_id] = {}
        self.store[user_id][key] = value
        self._save()

    def append_to_list(self, user_id: str, key: str, value: str):
        if user_id not in self.store:
            self.store[user_id] = {}
        if key not in self.store[user_id]:
            self.store[user_id][key] = []
        self.store[user_id][key].append(value)
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.store, f, indent=2)
