import numpy as np
from scipy.optimize import fsolve
from matplotlib import animation
import matplotlib.pyplot as plt
import copy


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


class Bead:
    def __init__(self, pos, theta, disp, w):
        self.pos = pos
        self.theta = theta
        self.disp = disp
        self.w = w
        self.D = np.array(self.pos) + np.array([r / np.sin(thtb) * np.cos(np.pi - thtb - self.theta),
                                                r / np.sin(thtb) * np.sin(np.pi - thtb - self.theta)])
        self.B = np.array(self.pos) + np.array([R / np.sin(thtb) * np.cos(np.pi - thtb - self.theta),
                                                R / np.sin(thtb) * np.sin(np.pi - thtb - self.theta)])
        self.F = np.array(self.pos) - np.array(
            [r / np.sin(thtb) * np.cos(thtb - self.theta), r / np.sin(thtb) * np.sin(thtb - self.theta)])
        self.H = np.array(self.pos) - np.array(
            [R / np.sin(thtb) * np.cos(thtb - self.theta), R / np.sin(thtb) * np.sin(thtb - self.theta)])
        self.A = np.array(self.B) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.C = np.array(self.D) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.E = np.array(self.F) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])
        self.G = np.array(self.H) - np.array([self.w * np.cos(-self.theta), self.w * np.sin(-self.theta)])


class String:
    def __init__(self, left_pos, right_pos, bead, out):
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.bead = bead
        self.out = out


def bead_pos(prev_bead, tht, disp):
    if disp >= 0:
        contact_pt = prev_bead.H
        back_corner = contact_pt - np.array([disp * np.cos(thtb - tht), disp * np.sin(thtb - tht)])
    else:
        back_corner = prev_bead.H + np.array([-disp * np.cos(thtb), -disp * np.sin(thtb)])
    front_corner = back_corner + np.array([w * np.cos(tht), -w * np.sin(tht)])
    pos = front_corner + np.array([(R / np.sin(thtb)) * np.cos(thtb - tht), (R / np.sin(thtb)) * np.sin(thtb - tht)])
    return pos


def connected(bead1, bead2):
    if bead1.theta == bead2.theta:
        return True
    else:
        return False


def intersect(bead_no, st_pt, out):
    if out:
        fst_pt_bot = beads[bead_no + 1].E
        fst_pt_top = beads[bead_no + 1].C
        scd_pt_top = beads[bead_no + 1].D
        scd_pt_bot = beads[bead_no + 1].F
    else:
        fst_pt_bot = beads[bead_no].F
        fst_pt_top = beads[bead_no].D
        scd_pt_top = beads[bead_no + 1].C
        scd_pt_bot = beads[bead_no + 1].E
    if scd_pt_top[0] <= fst_pt_top[0]:
        return 'bot'
    if st_pt[1] > fst_pt_top[1] + (fst_pt_top[1] - scd_pt_bot[1]) / (fst_pt_top[0] - scd_pt_bot[0]) * (
            st_pt[0] - fst_pt_top[0]):
        return 'top'
    elif st_pt[1] < fst_pt_bot[1] + (fst_pt_bot[1] - scd_pt_top[1]) / (fst_pt_bot[0] - scd_pt_top[0]) * (
            st_pt[0] - fst_pt_bot[0]):
        return 'bot'
    else:
        return 'none'


# fsolve equations
def surface(vars, contact_forces):
    x, y, z = vars
    sum_x = 0
    sum_y = 0
    sum_M = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
        sum_M += -force.mag * np.cos(force.ang) * force.ypos + force.mag * np.sin(force.ang) * force.xpos
    eq1 = x + sum_x
    eq2 = y + sum_y
    eq3 = z + sum_M
    return eq1, eq2, eq3


def point_contact(vars, contact_forces):
    x, y = vars
    # sum over x
    sum_x = 0
    sum_y = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
    eq1 = sum_x + x * np.cos(contact_forces[4].ang) + y * np.cos(contact_forces[5].ang)
    eq2 = sum_y + x * np.sin(contact_forces[4].ang) + y * np.sin(contact_forces[5].ang)
    return eq1, eq2


def two_point_contact(vars, contact_forces):
    x, y = vars
    sum_x = 0
    sum_y = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
    eq1 = sum_x + x * np.cos(contact_forces[4].ang) + y * np.cos(contact_forces[5].ang)
    eq2 = sum_y + x * np.sin(contact_forces[4].ang) + y * np.sin(contact_forces[5].ang)
    return eq1, eq2


def top_eqn(vars, z, surface_force):
    x = vars
    eq1 = z + np.cos(surface_force.ang) * (surface_force.ypos + x) * surface_force.mag - np.sin(
        surface_force.ang) * surface_force.mag * (surface_force.xpos - x / np.tan(thtb))
    return eq1


def bot_eqn(vars, z, surface_force):
    x = vars
    eq1 = z + np.cos(surface_force.ang) * (surface_force.ypos - x) * surface_force.mag - np.sin(
        surface_force.ang) * surface_force.mag * (surface_force.xpos - x / np.tan(thtb))
    return eq1


n = 100
param_sweep=3
n_iters = 100
n_beads = 1
bead_len = 5
r = 0.0003
R = 0.0075
w_b = 0.015
k = 10000
delt = 0.005
thtb = np.pi / 3
mu = 0.1
w = bead_len * w_b
d = (R) / np.sin(thtb)
tht = np.linspace(0.001, 1, n)
F = 30
ang = 0
disp = 0

unstable = True
beads = np.zeros(n_beads + 1, dtype=Bead)
strings = np.zeros((2 * n_beads + 2, 2))
str_type = np.zeros(2 * n_beads + 2, dtype=String)
all_forces = []
shapes = []
'''''''''''
Needs fixing, add condition for thtb>pi/4
'''''''''''


def min_ang(disp):
    ang3 = np.arcsin((d - disp) * np.sin(2 * thtb) / d)
    return np.pi - 2 * thtb - ang3


def max_disp(ang):
    ang3 = np.pi - ang - 2 * thtb
    if ang == 0:
        return 0
    if thtb == np.pi / 2:
        return R
    else:
        disp = d - np.sin(ang3) * d / np.sin(2 * thtb)
    return disp


fin_tht = np.zeros((n_iters, n_beads,param_sweep))
fin_disp = np.zeros((n_iters, n_beads,param_sweep))


def main():
    # tht_ini_ini = np.random.random_sample(n_beads) / n_beads
    tht_ini = [0.0] * n_beads
    # disp_ini = np.random.random_sample(n_beads) * 0.001
    disp_ini = [0.00]
    # initialize beads
    for j in range(param_sweep):
        contact_type = np.zeros(n_beads, dtype=String)
        delt=0.002*(j+1)

        for m in range(n_iters):
            F=m*0.5
            for i in range(n_beads):
                contact_type[n_beads - 1 - i] = 'surface'
            tht_ini = [0.0] * n_beads
            # disp_ini = np.random.random_sample(n_beads) * 0.001
            disp_ini = [0.00]
            beads[0] = Bead([0, 0], 0, 0, w_b)
            moment=1
            y_sum=1
            progress=0
            while np.abs(moment) > 0.025 or np.abs(y_sum) >0.1:
                for i in range(n_beads):
                    # must be difference of tht_ini values when more than one bead
                    if max_disp(tht_ini[i]) < disp_ini[i]:
                        disp_ini[i] = max_disp(tht_ini[i])
                for i in range(1, n_beads + 1):
                    new_pos = bead_pos(beads[i - 1], tht_ini[i - 1], disp_ini[i - 1])
                    new_bead = Bead(new_pos, tht_ini[i - 1], disp_ini[i - 1], w)
                    beads[i] = new_bead
                '''''''''
                flawed: change so the angles are rotated so the left one is 0
                '''''''''
                # initialize string points
                strings[0, :] = beads[0].pos[0] - w_b, beads[0].pos[1]
                str_type[0] = 'none'
                for i in range(2 * n_beads):
                    if i % 2 == 0:
                        out = False
                    else:
                        out = True
                    bead_no = np.floor(i / 2).astype(int)

                    st_pt = strings[i, :]
                    result = intersect(bead_no, st_pt, out)
                    if out:
                        if result == 'top':
                            strings[i + 1, :] = beads[bead_no + 1].C
                            str_type[i + 1] = 'top'
                        elif result == 'bot':
                            strings[i + 1, :] = beads[bead_no + 1].E
                            str_type[i + 1] = 'bot'
                        else:
                            strings[i + 1, :] = [beads[bead_no + 1].C[0], beads[bead_no + 1].C[1] - r]
                            str_type[i + 1] = 'none'

                    else:
                        if result == 'top':
                            strings[i + 1, :] = beads[bead_no].D
                            str_type[i + 1] = 'top'
                        elif result == 'bot':
                            strings[i + 1, :] = beads[bead_no].F
                            str_type[i + 1] = 'bot'
                        else:
                            strings[i + 1, :] = [beads[bead_no].D[0], beads[bead_no].D[1] - r]
                            str_type[i + 1] = 'none'
                strings[2 * n_beads + 1, :] = beads[n_beads].pos
                string_arr = np.vstack(strings)
                # calculate string length
                string_len = (-w * n_beads - w_b) + delt
                for i in range(2 * n_beads + 1):
                    dist = np.sqrt(
                        (string_arr[i, 0] - string_arr[i + 1, 0]) ** 2 + (string_arr[i, 1] - string_arr[i + 1, 1]) ** 2)
                    string_len += dist
                tension = Force(beads[n_beads].pos[0], beads[n_beads].pos[1], k * string_len, np.pi - beads[n_beads].theta)
                F_app = Force(0.5 * (beads[n_beads].A[0] + beads[n_beads].B[0]),
                              0.5 * (beads[n_beads].A[1] + beads[n_beads].B[1]),
                              F, 3 * np.pi / 2)
                # calculate string forces
                forces_str = []
                for i in range(1, 2 * n_beads + 1):
                    vertex = string_arr[i, :]
                    left = string_arr[i - 1, :]
                    right = string_arr[i + 1, :]
                    left_vec = left - vertex
                    right_vec = right - vertex
                    norm_left = left_vec / np.linalg.norm(left_vec)
                    norm_right = right_vec / np.linalg.norm(right_vec)
                    force_sum = norm_left * tension.mag + norm_right * tension.mag
                    force_ang_left = np.arcsin(norm_left[1])
                    if norm_left[0] < 0:
                        force_ang_left = np.pi - force_ang_left
                    force_ang_right = np.arcsin(norm_right[1])
                    if norm_right[0] < 0:
                        force_ang_right = np.pi - force_ang_right
                    while force_ang_right < 0 or force_ang_right > 2 * np.pi or force_ang_left < 0 or force_ang_left > 2 * np.pi:
                        if force_ang_right < 0:
                            force_ang_right += 2 * np.pi
                        if force_ang_left < 0:
                            force_ang_left += 2 * np.pi
                        if force_ang_right > 2 * np.pi:
                            force_ang_right -= 2 * np.pi
                        if force_ang_left > 2 * np.pi:
                            force_ang_left -= 2 * np.pi
                    if force_ang_right - force_ang_left < np.pi:
                        force_ang = 0.5 * (force_ang_right + force_ang_left)
                    else:
                        force_ang = 0.5 * (force_ang_right + force_ang_left) - np.pi
                    if str_type[i] == 'none':
                        force_sum = [0, 0]
                    forces_str.append(Force(vertex[0], vertex[1], np.linalg.norm(force_sum), force_ang))

                # calculate forces
                for i in range(n_beads):
                    bead = beads[n_beads - i]
                    contact_forces = []
                    if i == 0:
                        right_force_1 = tension
                        right_force_2 = F_app
                        str_force_r = Force(beads[n_beads].pos[0], beads[n_beads].pos[1], 0, np.pi - beads[n_beads].theta)
                    else:
                        right_force_1 = copy.copy(left_force_1)
                        right_force_2 = copy.copy(left_force_2)
                        str_force_r = forces_str[2 * n_beads - 2 * i]
                        if right_force_1.ang > np.pi:
                            right_force_1.ang -= np.pi
                        else:
                            right_force_1.ang += np.pi
                        if right_force_2.ang > np.pi:
                            right_force_2.ang -= np.pi
                        else:
                            right_force_2.ang += np.pi
                    str_force_l = forces_str[2 * n_beads - 1 - 2 * i]
                    contact_forces.append(right_force_1)
                    contact_forces.append(right_force_2)
                    contact_forces.append(str_force_r)
                    contact_forces.append(str_force_l)

                    # string forces

                    if contact_type[n_beads - i - 1] == 'surface':
                        # contact surface
                        x, y, z = fsolve(surface, (1, 1, 1), contact_forces)
                        y = x * np.tan((max(min(thtb, np.arctan(y / x)), -thtb)))
                        surface_force = Force(beads[n_beads - i - 1].pos[0], beads[n_beads - i - 1].pos[1],
                                              np.sqrt(x ** 2 + y ** 2), np.arctan(y / x))
                        dummy_force = Force(beads[n_beads - i - 1].pos[0], beads[n_beads - i - 1].pos[1], 0,
                                            np.arctan(y / x))

                        if z < surface_force.ypos * surface_force.mag * np.sin(
                                surface_force.ang) - surface_force.xpos * surface_force.mag * np.cos(surface_force.ang):
                            a = fsolve(top_eqn, (.01), (z, surface_force))  # a is the vertical distance from the middle
                            a = a[0]
                            if a > R:
                                a = R
                            surface_force = Force(beads[n_beads - i - 1].pos[0] - a / np.tan(thtb),
                                                  beads[n_beads - i - 1].pos[1] + a,
                                                  np.sqrt(x ** 2 + y ** 2), np.arctan(y / x))
                        else:
                            a = fsolve(bot_eqn, (0.01), (z, surface_force))
                            a = a[0]
                            if a > R:
                                a = R
                            surface_force = Force(beads[n_beads - i - 1].pos[0] - a / np.tan(thtb),
                                                  beads[n_beads - i - 1].pos[1] - a,
                                                  np.sqrt(x ** 2 + y ** 2), np.arctan(y / x))

                        contact_forces.append(surface_force)
                        contact_forces.append(dummy_force)
                        moment = 0
                        y_sum = 0
                        for force in contact_forces:
                            moment += -force.mag * np.cos(force.ang) * force.ypos + force.mag * force.xpos * np.sin(
                                force.ang)
                            y_sum += force.mag * np.sin(force.ang)
                        tht_ini[n_beads - i - 1] -= max(min(0.1, moment / 100), -0.1)
                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                        if i == n_beads - 1:
                            out_ang = thtb
                        else:
                            out_ang = tht_ini[n_beads - i - 2] + thtb
                        across_ang = np.pi - thtb - tht_ini[n_beads - i - 1]
                        if out_ang + np.pi / 2 <= across_ang:
                            contact_type[n_beads - i - 1] = 'two'
                            # should be difference of angles for multiple beads
                            disp_ini[n_beads - i - 1] = max_disp(tht_ini[n_beads - i - 1])
                        else:
                            contact_type[n_beads - i - 1] = 'one'
                            disp_ini[n_beads - i - 1] -= max(min(0.0002, y_sum / 100000), -0.0002)
                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                            moment=0
                        left_force_1 = surface_force
                        left_force_2 = dummy_force

                    elif contact_type[n_beads - i - 1] == 'two':
                        # 2 contact points
                        if disp_ini[n_beads - i - 1] >= 0:
                            force_top = Force(beads[n_beads - i - 1].F[0], beads[n_beads - i - 1].F[1], 0,
                                              np.pi - thtb - bead.theta - np.pi / 2 + np.arctan(mu))
                            force_bottom = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                                 np.pi + thtb - bead.theta + np.pi / 2 + np.arctan(mu))
                        else:
                            force_top = Force(beads[n_beads - i - 1].F[0], beads[n_beads - i - 1].F[1], 0,
                                              np.pi - thtb - bead.theta - np.pi / 2 + np.arctan(mu))
                            force_bottom = Force(beads[n_beads - i].G[0], beads[n_beads - i].G[1], 0,
                                                 np.pi + thtb - beads[n_beads - i - 1].theta + np.pi / 2 + np.arctan(mu))
                        contact_forces.append(force_top)
                        contact_forces.append(force_bottom)
                        x, y = fsolve(two_point_contact, (1, 1), contact_forces)
                        if x <= 0: x = 0
                        if y <= 0: y = 0
                        force_top.mag = x
                        force_bottom.mag = y
                        moment = 0
                        y_sum = 0
                        for force in contact_forces:
                            moment += -force.mag * np.cos(force.ang) * force.ypos + force.mag * force.xpos * np.sin(
                                force.ang)
                            y_sum += force.mag * np.sin(force.ang)
                        tht_ini[n_beads - i - 1] -= max(min(0.1, moment / 100), -0.1)
                        if moment>0.01 and disp_ini[n_beads-i-1]>max_disp(tht_ini[n_beads-i-1]):
                            disp_ini[n_beads - i - 1] = max_disp(tht_ini[n_beads - i - 1])

                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                        if i == n_beads - 1:
                            out_ang = thtb
                        else:
                            out_ang = tht_ini[n_beads - i - 2] + thtb
                        across_ang = np.pi - thtb - tht_ini[n_beads - i - 1]
                        if out_ang + np.pi / 2 <= across_ang:
                            contact_type[n_beads - i - 1] = 'two'
                            # should be diff of angles
                            disp_ini[n_beads - i - 1] = max_disp(tht_ini[n_beads - i - 1])
                        else:
                            contact_type[n_beads - i - 1] = 'one'
                            disp_ini[n_beads - i - 1] -= max(min(0.0002, y_sum / 100000), -0.0002)
                            if tht_ini[n_beads-i-1]<min_ang(disp_ini[n_beads-i-1]):
                                tht_ini[n_beads - i - 1] = min_ang(disp_ini[n_beads - i - 1])
                                contact_type[n_beads-i-1] = 'two'
                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                            moment=0
                        left_force_1 = force_top
                        left_force_2 = force_bottom

                    else:
                        # one contact point
                        if disp >= 0:
                            forces_norm = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                                thtb - np.pi / 2 - beads[n_beads - i - 1].theta)
                            forces_fric = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                                thtb - beads[n_beads - i - 1].theta)
                        else:
                            forces_norm = Force(beads[n_beads - i].G[0], beads[i + 1].G[1], 0,
                                                thtb - np.pi / 2 - beads[n_beads - i - 1].theta)
                            forces_fric = Force(beads[n_beads - i].G[0], beads[i + 1].G[1], 0,
                                                thtb - beads[n_beads - i - 1].theta)
                        contact_forces.append(forces_norm)
                        contact_forces.append(forces_fric)
                        x, y = fsolve(point_contact, (1, 1), contact_forces)
                        if y > x * mu:
                            y = x * mu
                        forces_norm.mag = x
                        forces_fric.mag = y
                        moment = 0
                        y_sum = 0
                        for force in contact_forces:
                            moment += -force.mag * np.cos(force.ang) * force.ypos + force.mag * force.xpos * np.sin(
                                force.ang)
                            y_sum += force.mag * np.sin(force.ang)
                        tht_ini[n_beads - i - 1] -= max(min(0.1, moment / 100), -0.1)
                        if moment>0.01 and disp_ini[n_beads-i-1]>max_disp(tht_ini[n_beads-i-1]):
                            disp_ini[n_beads - i - 1] = max_disp(tht_ini[n_beads - i - 1])

                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                        if i == n_beads - 1:
                            out_ang = thtb
                        else:
                            out_ang = tht_ini[n_beads - i - 2] + thtb
                        across_ang = np.pi - thtb - tht_ini[n_beads - i - 1]
                        if out_ang + np.pi / 2 <= across_ang:
                            contact_type[n_beads - i - 1] = 'two'
                            # should be diff of angles
                            disp_ini[n_beads - i - 1] = max_disp(tht_ini[n_beads - i - 1])
                        else:
                            contact_type[n_beads - i - 1] = 'one'
                            disp_ini[n_beads - i - 1] -= max(min(0.0002, y_sum / 100000), -0.0002)
                            if tht_ini[n_beads-i-1]<min_ang(disp_ini[n_beads-i-1]):
                                tht_ini[n_beads - i - 1] = 0.5*(min_ang(disp_ini[n_beads - i - 1])+tht_ini[n_beads-i-1])
                                disp_ini[n_beads-i-1] = max_disp(tht_ini[n_beads-i-1])
                                contact_type[n_beads-i-1] = 'two'
                        if tht_ini[n_beads - i - 1] <= 0:
                            tht_ini[n_beads - i - 1] = 0
                            contact_type[n_beads-i-1] = 'surface'
                            moment=0
                        left_force_1 = forces_norm
                        left_force_2 = forces_fric

                    if i == 0:
                        all_forces = []
                    all_forces.append(right_force_1)
                    all_forces.append(right_force_2)
                    all_forces.append(str_force_l)
                    all_forces.append(str_force_r)
                    all_forces.append(left_force_1)
                    all_forces.append(left_force_2)
                progress+=1
                if progress>=1000:
                    print(moment)
                    print(y_sum)
                    print(m)
                    break
            if m % 99 == 100:

                shapes = []
                # for this bead and all beads to the right
                # add displacement, angle around the contact point
                for bead in beads:
                    top = [bead.A, bead.B, bead.D, bead.C]
                    bot = [bead.E, bead.F, bead.H, bead.G]
                    top_poly = plt.Polygon(top)
                    bot_poly = plt.Polygon(bot)
                    shapes.append(top_poly)
                    shapes.append(bot_poly)
                plt.axes()

                for shape in shapes:
                    plt.gca().add_patch(shape)
                    count = -1
                for force in all_forces:
                    count += 1
                    if force.mag == 0:
                        continue
                    if np.floor(count / 6) % 3 == 0:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 10000,
                                        force.mag * np.sin(force.ang) / 10000, color='black', width=0.0005)
                    elif np.floor(count / 6) % 3 == 1:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 10000,
                                        force.mag * np.sin(force.ang) / 10000, color='orange', width=0.0005)
                    else:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 10000,
                                        force.mag * np.sin(force.ang) / 10000, color='green', width=0.0005)
                    plt.gca().add_patch(bow)
                plt.axis('scaled')
                plt.plot(string_arr[:, 0], string_arr[:, 1], color='red')
                plt.show()
            for i in range(n_beads):
                fin_disp[m, i, j] = disp_ini[i]
                fin_tht[m, i, j] = tht_ini[i]

    for i in range(param_sweep):
        plt.plot(fin_disp[:,0,i])
    plt.show()
    for i in range(param_sweep):
        plt.plot(fin_tht[:,0,i])
    plt.show()



main()
