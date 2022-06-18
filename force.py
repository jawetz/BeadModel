import numpy as np
class Force:
    def __init__(self, xpos, ypos, mag, ang):
        self.xpos = xpos
        self.ypos = ypos
        self.mag = mag
        self.ang = ang

    def momentAt(self, x, y):
        xdis = self.xpos - x
        ydis = self.ypos - y
        M = self.mag * xdis * np.sin(self.ang) - (self.mag * ydis * np.cos(self.ang))
        return M
