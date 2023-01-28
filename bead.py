import numpy as np

class Bead:
    def __init__(self, pos, theta, disp, w, R, r, thtb):
        self.pos = pos
        self.theta = theta
        self.disp = disp
        self.w = w
        self.R=R
        self.r=r
        self.thtb=thtb
        self.vol=np.pi*(self.R**2)*self.w
        self.D = np.array(self.pos) + np.array([self.r / np.sin(self.thtb) * np.cos(np.pi - self.thtb - self.theta),
                                                self.r / np.sin(self.thtb) * np.sin(np.pi - self.thtb - self.theta)])
        self.B = np.array(self.pos) + np.array([self.R / np.sin(self.thtb) * np.cos(np.pi - self.thtb - self.theta),
                                                self.R / np.sin(self.thtb) * np.sin(np.pi - self.thtb - self.theta)])
        self.F = np.array(self.pos) - np.array(
            [self.r / np.sin(self.thtb) * np.cos(self.thtb - self.theta), self.r / np.sin(self.thtb) * np.sin(self.thtb - self.theta)])
        self.H = np.array(self.pos) - np.array(
            [self.R / np.sin(self.thtb) * np.cos(self.thtb - self.theta), self.R / np.sin(self.thtb) * np.sin(self.thtb - self.theta)])
        self.A = np.array(self.B) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.C = np.array(self.D) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.E = np.array(self.F) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.G = np.array(self.H) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
