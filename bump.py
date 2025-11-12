import numpy as np

class Bump:
    # (pos, half_width, height)
    _original_multi_specs = np.array([
        (-30.0, 1.0, 0.1),
        (10.0, 1.0, 0.1),
        (50.0, 1.0, 0.2),
        (90.0, 1.5, 0.05),
        (130.0, 2.0, 0.05),
        (170.0, 2.5, 0.1),
        (210.0, 1.2, 0.1),
        (250.0, 1.0, 0.2),
        (290.0, 1.5, 0.05),
        (330.0, 2.0, 0.05),
        (370.0, 2.5, 0.1),
        (410.0, 1.2, 0.1),
        (450.0, 1.0, 0.2),
        (490.0, 1.5, 0.05),
        (530.0, 2.0, 0.05),
        (570.0, 2.5, 0.1),
        (610.0, 1.2, 0.1),
    ])
    original_single_specs = np.array([
        (20, 1.25, 0.09),
    ])

    def __init__(self, is_multi_bump=False):
        self.is_multi_bump = is_multi_bump
        if is_multi_bump:
            Bump._original_specs = Bump._original_multi_specs
        else:
            Bump._original_specs = Bump.original_single_specs
        self.base_specs = Bump._original_specs.copy()
        self.bump_specs = self.base_specs.copy()

    def set_rng(self, rng):
        self.rng = rng

    def reset(self):
        if self.is_multi_bump:
            self.reset_multi()
        else:
            self.reset_single()

    def reset_single(self):
        new_specs = np.copy(self.base_specs)
        new_specs[0, 1] += self.rng.uniform(-0.75, 0.75)
        new_specs[0, 2] += self.rng.uniform(-0.03, 0.03)
        self.bump_specs = new_specs

    def reset_multi(self):
        new_specs = []
        for pos, hw, h in self.base_specs:
            pos += self.rng.uniform(-5, 5)
            hw += self.rng.uniform(-0.1, 0.1)
            h += self.rng.uniform(-0.02, 0.02)
            new_specs.append((pos, hw, h))
        self.bump_specs = np.array(new_specs)            

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