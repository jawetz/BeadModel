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
    def __init__(self, pos, theta, disp):
        self.pos = pos
        self.theta = theta
        self.disp = disp
        self.D = np.array(self.pos) + np.array([r / np.sin(thtb) * np.cos(np.pi - thtb - self.theta),
                                                r / np.sin(thtb) * np.sin(np.pi - thtb - self.theta)])
        self.B = np.array(self.pos) + np.array([R / np.sin(thtb) * np.cos(np.pi - thtb - self.theta),
                                                R / np.sin(thtb) * np.sin(np.pi - thtb - self.theta)])
        self.F = np.array(self.pos) - np.array(
            [r / np.sin(thtb) * np.cos(thtb - self.theta), r / np.sin(thtb) * np.sin(thtb - self.theta)])
        self.H = np.array(self.pos) - np.array(
            [R / np.sin(thtb) * np.cos(thtb - self.theta), R / np.sin(thtb) * np.sin(thtb - self.theta)])
        self.A = np.array(self.B) - np.array([w * np.cos(-self.theta), w * np.sin(-self.theta)])
        self.C = np.array(self.D) - np.array([w * np.cos(-self.theta), w * np.sin(-self.theta)])
        self.E = np.array(self.F) - np.array([w * np.cos(-self.theta), w * np.sin(-self.theta)])
        self.G = np.array(self.H) - np.array([w * np.cos(-self.theta), w * np.sin(-self.theta)])


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
        back_corner = prev_bead.H + np.array([disp * np.cos(thtb), disp * np.sin(thtb)])
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
    sum_M = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_M += -force.mag * np.cos(force.ang) * force.ypos + force.mag * np.sin(force.ang) * force.xpos
    eq1 = sum_x + x * np.cos(contact_forces[4].ang) + y * np.cos(contact_forces[5].ang)
    eq2 = sum_M - x * np.cos(contact_forces[4].ang) * contact_forces[4].ypos + x * np.sin(contact_forces[4].ang) * \
          contact_forces[4].xpos - y * np.cos(contact_forces[5].ang) * contact_forces[5].ypos + y * np.sin(
        contact_forces[5].ang) * contact_forces[5].xpos
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


beads = []
strings = []
str_type = []
forces_norm = []
forces_fric = []
all_forces = []
n = 100
n_beads = 2
r = 0.0003
R = 0.0075
w = 0.015
k = 10000
delt = 0.00
thtb = np.pi / 4
mu = 0.1
len = w / 2
d = (R) / np.sin(thtb)
tht = np.linspace(0.001, 1, n)
F = 20

unstable = True

shapes = []
'''''''''''
Needs fixing, add condition for thtb>pi/4
'''''''''''


def min_ang(disp):
    ang3 = np.arcsin((d - disp) * np.sin(2 * thtb) / d)
    return np.pi - 2 * thtb - ang3


def work(tht_ini, disp_ini):

    for i in range(1, n_beads + 1):
        new_pos = bead_pos(beads[i - 1], tht_ini[i - 1], disp_ini[i - 1])
        new_bead = Bead(new_pos, tht_ini[i - 1], disp_ini[i - 1])
        beads.append(new_bead)
    '''''''''
    flawed: change so the angles are rotated so the left one is 0
    '''''''''
    # initialize string points
    strings.append(np.array([beads[0].pos[0] - w, beads[0].pos[1]]))
    str_type.append('none')
    for i in range(2 * n_beads):
        if i % 2 == 0:
            out = False
        else:
            out = True
        bead_no = np.floor(i / 2).astype(int)
        st_pt = strings[-1]
        result = intersect(bead_no, st_pt, out)
        if out:
            if result == 'top':
                strings.append(beads[bead_no + 1].C)
                str_type.append('top')
            elif result == 'bot':
                strings.append(beads[bead_no + 1].E)
                str_type.append('bot')
            else:
                strings.append([beads[bead_no + 1].C[0], beads[bead_no + 1].C[1] - r])
                str_type.append('none')

        else:
            if result == 'top':
                strings.append(beads[bead_no].D)
                str_type.append('top')
            elif result == 'bot':
                strings.append(beads[bead_no].F)
                str_type.append('bot')
            else:
                strings.append([beads[bead_no].D[0], beads[bead_no].D[1] - r])
                str_type.append('none')
    strings.append(np.array([beads[n_beads].pos]))
    string_arr = np.vstack(strings)
    # calculate string length
    string_len = (-w * (n_beads + 1)) + delt
    for i in range(2 * n_beads + 1):
        dist = np.sqrt((string_arr[i, 0] - string_arr[i + 1, 0]) ** 2 + (string_arr[i, 1] - string_arr[i + 1, 1]) ** 2)
        string_len += dist
    tension = Force(beads[n_beads].pos[0], beads[n_beads].pos[1], k * string_len, np.pi - beads[n_beads].theta)
    F_app = Force(0.5 * (beads[n_beads].A[0] + beads[n_beads].B[0]), 0.5 * (beads[n_beads].A[1] + beads[n_beads].B[1]),
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

        if bead.disp == 0:
            # contact surface
            x, y, z = fsolve(surface, (1, 1, 1), contact_forces)
            surface_force = Force(beads[n_beads - i - 1].pos[0], beads[n_beads - i - 1].pos[1],
                                  np.sqrt(x ** 2 + y ** 2), np.arctan(y / x))
            dummy_force = Force(beads[n_beads - i - 1].pos[0], beads[n_beads - i - 1].pos[1], 0, np.arctan(y / x))

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
            left_force_1 = surface_force
            left_force_2 = dummy_force

        elif beads[n_beads - i].theta - beads[n_beads - i - 1].theta <= min_ang(disp_ini[n_beads - i - 1]):
            # 2 contact points
            force_top = Force(beads[n_beads - i - 1].F[0], beads[n_beads - i - 1].F[1], 0,
                              np.pi - thtb - bead.theta - np.pi / 2)
            force_bottom = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                 np.pi + thtb - bead.theta + np.pi / 2)
            contact_forces.append(force_top)
            contact_forces.append(force_bottom)
            x, y = fsolve(two_point_contact, (1, 1), contact_forces)
            force_top.mag = x
            force_bottom.mag = y
            left_force_1 = force_top
            left_force_2 = force_bottom

        else:
            # one contact point

            forces_norm = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                thtb - np.pi / 2 - beads[n_beads - i - 1].theta)
            forces_fric = Force(beads[n_beads - i - 1].H[0], beads[n_beads - i - 1].H[1], 0,
                                thtb - beads[n_beads - i - 1].theta)
            contact_forces.append(forces_norm)
            contact_forces.append(forces_fric)
            x, y = fsolve(point_contact, (1, 1), contact_forces)
            if y > x * mu:
                y = x * mu
            forces_norm.mag = x
            forces_fric.mag = y

            left_force_1 = forces_norm
            left_force_2 = forces_fric

        current_forces = []
        current_forces.append(right_force_1)
        current_forces.append(right_force_2)
        current_forces.append(str_force_l)
        current_forces.append(str_force_r)
        current_forces.append(left_force_1)
        current_forces.append(left_force_2)

        moment = 0
        y_sum = 0
        for force in current_forces:
            moment += -force.mag * np.cos(force.ang) * force.ypos + force.mag * force.xpos * np.sin(force.ang)
            y_sum += force.mag * np.sin(force.ang)
        if y_sum<0 and disp_ini[n_beads-i-1]<=d:
            disp_ini[n_beads-i-1]+=0.0001
        if moment<0:
            tht_ini[n_beads-i-1]+=0.01
        all_forces.append(right_force_1)
        all_forces.append(right_force_2)
        all_forces.append(str_force_l)
        all_forces.append(str_force_r)
        all_forces.append(left_force_1)
        all_forces.append(left_force_2)
        #save bead states, string states

    return tht_ini, disp_ini, all_forces
        # for this bead and all beads to the right
        # add displacement, angle around the contact point


def show(beads,all_forces, string_arr):
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


def main():
    beads.append(Bead([0, 0], 0, 0))
    # tht_ini_ini = np.random.random_sample(n_beads) / n_beads
    tht_ini_ini = [0] * n_beads
    # disp_ini = np.random.random_sample(n_beads) * 0.001
    disp_ini = [0, 0.001]
    # initialize beads
    for i in range(n_beads):
        if min_ang(disp_ini[i]) > tht_ini_ini[i]:
            tht_ini_ini[i] = min_ang(disp_ini[i])
    tht_ini = np.cumsum(tht_ini_ini)
    new_tht, new_disp, all_forces = work(tht_ini, disp_ini)
    for i in range(100):
        new_tht,new_disp, all_forces=work(new_tht,new_disp)

main()
