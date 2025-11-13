import json, os, numpy as np

class HistoryLib:
    def __init__(self, path="outputs/history_lib.json"):
        self.path = path
        self.db = {"tasks": []}
        if os.path.exists(path):
            with open(path, "r") as f:
                self.db = json.load(f)

    def add_task(self, name, X, y):
        self.db["tasks"].append({"name": name, "X": np.asarray(X).tolist(), "y": np.asarray(y).tolist()})
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.db, f, indent=2)

    def load_all(self):
        tasks = []
        for t in self.db["tasks"]:
            tasks.append((t["name"], np.array(t["X"]), np.array(t["y"])))
        return tasks
