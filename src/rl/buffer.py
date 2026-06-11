class RolloutBuffer:
    def __init__(self):

        self.X = []
        self.dynamic_edges = []
        self.static_edges = []

        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

    def clear(self):

        self.X.clear()
        self.dynamic_edges.clear()
        self.static_edges.clear()

        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()