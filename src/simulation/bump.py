import numpy as np

class Bump:
    # (pos, half_width, height)
    _original_specs = np.array([
        (3, 1.25, 0.09),
    ])

    def __init__(self):
        self.base_specs = Bump._original_specs.copy()
        self.bump_specs = self.base_specs.copy()

    def reset(self):
        self.bump_specs = self.base_specs.copy()

    def customize(self, bump_specs):
        self.bump_specs = np.array(bump_specs)

    @staticmethod
    def _bump_shape(dist, half_width, height):  # sinusoidal
        return height * (1 + np.cos(np.pi * dist / half_width)) / 2

    def __call__(self, x):
        for pos, hw, h in self.bump_specs:
            d = x - pos
            if -hw <= d <= hw:
                return self._bump_shape(d, hw, h)
        return 0.0

