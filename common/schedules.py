class LinearSchedule(object):
    def __init__(self, start, end, steps):
        self.steps = steps
        self.start = start
        self.end = end

    def value(self, t):
        fraction = min(float(t) / self.steps, 1.0)
        return self.start + fraction * (self.end - self.start)
