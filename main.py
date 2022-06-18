import numpy as np
from scipy.optimize import fsolve
from matplotlib import animation
import matplotlib.pyplot as plt
import copy
import pandas
from force import Force
from bead import Bead
from bead_pos import bead_pos
from intersect import intersect
from contacts import surface_contact, point_contact, two_point_contact, positioning, friction
from no_overlap import max_disp, min_ang
from string_crt_pt import two_pt_crt_pt
from k_calc import k_calc
import argparse


def init():
    shapes.set_data([], [])
    return shapes


def animate(i):
    shapes = []  # for making the pictures
    # for this bead and all beads to the right
    # add displacement, angle around the contact point
    beads[0] = Bead([0, 0], 0, 0, 0.015, args.R, args.r, args.thtb,args.mis)
    for j in range(args.n_beads):
        new_pos = bead_pos(beads[j], fin_tht[i, j, 0], fin_disp[i, j, 0], args)  # adds beads to the system
        new_bead = Bead(new_pos, fin_tht[i, j, 0], fin_disp[i, j, 0], args.w, args.R, args.r, args.thtb,args.mis)
        beads[j + 1] = new_bead
    for bead in beads:
        top = [bead.A, bead.B, bead.D, bead.C]
        bot = [bead.E, bead.F, bead.H, bead.G]
        top_poly = plt.Polygon(top)
        bot_poly = plt.Polygon(bot)
        shapes.append(top_poly)
        shapes.append(bot_poly)  # makes a list of shapes

    for shape in shapes:
        plt.gca().add_patch(shape)  # plots shapes
    plt.xlim([-0.05, 0.1])
    plt.ylim([-0.06, 0.03])
    return shapes


parser = argparse.ArgumentParser(description='Bead Model')
parser.add_argument('--n_iters', type=int, default=100, help='granularity of simulation')
parser.add_argument('--param_sweep', type=int, default=3, help='for varying parameters across trials')
parser.add_argument('--n_beads', type=int, default=10, help='number of non-fixed beads in the system')
parser.add_argument('--R', type=float, default=0.0075, help='outer radius')
parser.add_argument('--r', type=float, default=0.0001, help='inner bead radius')
parser.add_argument('--w', type=float, default=0.015, help='bead length')
parser.add_argument('--mis', type=float, default=np.pi/360, help='angle disparity')
parser.add_argument('--ys', type=float, default=145.58, help='youngs modulus times cross-sectional area')
parser.add_argument('--delt', type=float, default=0.003, help='initial displacement of string')
parser.add_argument('--thtb', type=float, default= 4*np.pi / 18, help='bead angle')
parser.add_argument('--mu', type=float, default=0.1, help='friction coefficient')
parser.add_argument('--grav_const', type=float, default=9.81, help='gravitational constant')
parser.add_argument('--dens', type=float, default=00, help='density of the material')
parser.add_argument('--tens',type=float,default=80,help='initial tension')
args = parser.parse_args()
steel_cable = pandas.read_csv(r"Steel_cable_test.csv").to_numpy()
force = (float(steel_cable[2265, 2]) - float(steel_cable[2265, 1]))
x_disp = (float(steel_cable[2265, 1]) - float(steel_cable[1532, 1]))

beads = np.zeros(args.n_beads + 1, dtype=Bead)
strings = np.zeros((2 * args.n_beads + 2, 2))
str_type = np.empty(2 * args.n_beads + 2, dtype=np.dtype('U100'))
all_forces = []
shapes = []
tht = np.linspace(0.001, 1, args.n_iters)

fin_tht = np.zeros((args.n_iters, args.n_beads, args.param_sweep))
fin_disp = np.zeros((args.n_iters, args.n_beads, args.param_sweep))
extension = np.zeros((args.n_iters, args.param_sweep))


def main():
    # initialize beads
    for j in range(args.param_sweep):
        contact_type = np.empty(args.n_beads, dtype=np.dtype('U100'))
        args.tens=10+40*j
        d = (args.R) / np.sin(args.thtb)  # length of angled side
        args.delt, k = k_calc(args)
        args.delt /= 1000
        print(k)
        # tht_ini_ini = np.random.random_sample(args.n_beads) / args.n_beads
        tht_ini=[0.00]*args.n_beads
        # disp_ini = np.random.random_sample(args.n_beads) * 0.001
        disp_ini = [0.00] * args.n_beads
        disp_ini[0] = max_disp(tht_ini[0], args)
        for i in range(args.n_beads):
            contact_type[args.n_beads - 1 - i] = 'two'
        contact_type[0] = 'two'
        finit = False
        for m in range(args.n_iters):
            global F
            F = m / args.n_iters * 6  # applied force
            # initial bead interface
            # disp_ini = np.random.random_sample(args.n_beads) * 0.001
            beads[0] = Bead([0, 0], 0, 0, 0.015, args.R, args.r, args.thtb,args.mis)  # fixed bead
            moment = np.ones(args.n_beads)
            y_sum = np.ones(args.n_beads)
            progress = 0
            while (np.amax(np.abs(moment)) > 0.001 or np.amax(
                    np.abs(y_sum)) > 0.0005) and finit == False:  # loops until convergence
                moment = np.zeros_like(moment)
                y_sum = np.zeros_like(y_sum)
                args.mu=0
                for i in range(args.n_beads):
                    # must be difference of tht_ini values when more than one bead
                    if max_disp(tht_ini[i], args) < disp_ini[i]:
                        if i == 0:
                            disp_ini[i] = max_disp(tht_ini[i], args)  # ensures initial beads dont overlap
                        else:
                            disp_ini[i] = max_disp(tht_ini[i] - tht_ini[i - 1], args)

                for i in range(1, args.n_beads + 1):
                    new_pos = bead_pos(beads[i - 1], tht_ini[i - 1], disp_ini[i - 1], args)  # adds beads to the system
                    new_bead = Bead(new_pos, tht_ini[i - 1], disp_ini[i - 1], args.w, args.R, args.r, args.thtb,args.mis)
                    beads[i] = new_bead

                # initialize string points
                str_type = np.empty(2 * args.n_beads + 2, dtype=np.dtype('U100'))
                strings[0, :] = beads[0].pos[0] - 0.230, beads[0].pos[1]  # initializes string
                str_type[0] = "start"
                strings[2 * args.n_beads + 1, :] = beads[args.n_beads].pos
                str_type[2 * args.n_beads + 1] = "end"
                st_pt = strings[0, :]
                for i in range(2 * args.n_beads):
                    if i % 2 == 0:
                        out = False  # whether string is going in or out of a bead
                    else:
                        out = True
                    bead_no = np.floor(i / 2).astype(int)
                    result = intersect(bead_no, st_pt, out, beads, args)
                    if out:
                        if result == "top":  # adds string points by where they contact the bead
                            strings[i + 1, :] = beads[bead_no + 1].C
                            str_type[i + 1] = "top"
                            st_pt = beads[bead_no + 1].C
                        elif result == "bot":
                            strings[i + 1, :] = beads[bead_no + 1].E
                            str_type[i + 1] = "bot"
                            st_pt = beads[bead_no + 1].E
                        else:
                            strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                            str_type[i + 1] = "none"
                    else:
                        if result == "top":
                            strings[i + 1, :] = beads[bead_no].D
                            str_type[i + 1] = "top"
                            st_pt = beads[bead_no].D
                        elif result == "bot":
                            strings[i + 1, :] = beads[bead_no].F
                            str_type[i + 1] = "bot"
                            st_pt = beads[bead_no].F
                        else:
                            strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                            str_type[i + 1] = "none"
                        if np.floor(i / 2).astype(int) == 0:
                            if tht_ini[np.floor(i / 2).astype(int)] == 0:
                                strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                                str_type[i + 1] = "none"
                        else:
                            if tht_ini[np.floor(i / 2).astype(int)] == tht_ini[np.floor(i / 2).astype(int) - 1]:
                                strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                                str_type[i + 1] = "none"

                for s in range(len(str_type)):
                    if str_type[s] == "none" and str_type[s - 1] != "none":
                        #keeps string straight when it is not touching a bead
                        left_pt = strings[s - 1, :]
                        t = 0
                        for t in range(s + 1, len(str_type)):
                            if str_type[t] != "none":
                                break
                        right_pt = strings[t, :]
                        for u in range(s, t):
                            strings[u] = ((u - s + 1) / (t - s + 1)) * (right_pt - left_pt) + (left_pt)
                global string_arr
                string_arr = np.vstack(strings)
                # calculate string length
                string_len = (-args.w * (args.n_beads) - 0.230) + args.delt  # initial string length
                for i in range(2 * args.n_beads + 1):
                    dist = np.sqrt(
                        (string_arr[i, 0] - string_arr[i + 1, 0]) ** 2 + (string_arr[i, 1] - string_arr[i + 1, 1]) ** 2)
                    string_len += dist  # stretched string length
                tension = Force(beads[args.n_beads].pos[0], beads[args.n_beads].pos[1], max(args.tens, 0),
                                np.pi - beads[args.n_beads].theta)
                F_app = Force(args.w * (args.n_beads), 0,
                              F, 3 * np.pi / 2)  # adds tension, applied force
                # calculate string forces
                forces_str = []
                # need to come up with exceptions for no contact case
                for i in range(1, 2 * args.n_beads + 1):  # calculates string-bead contact
                    flag = 'contact'
                    vertex = string_arr[i, :]
                    left = string_arr[i - 1, :]
                    right = string_arr[i + 1, :]
                    left_vec = left - vertex
                    right_vec = right - vertex
                    if np.linalg.norm(left_vec) <= 1e-9:
                        flag = 'no contact'
                    else:
                        norm_left = left_vec / np.linalg.norm(left_vec)  # points in the two directions
                    if np.linalg.norm(right_vec) <= 1e-9:
                        flag = 'no contact'
                    else:
                        norm_right = right_vec / np.linalg.norm(right_vec)
                    if flag == 'no contact':
                        force_sum = [0, 0]
                        force_ang = 10
                    else:
                        force_sum = norm_left * tension.mag + norm_right * tension.mag  # vector sum of tension forces
                        force_ang_left = np.arcsin(norm_left[1])
                        if norm_left[0] < 0:
                            force_ang_left = np.pi - force_ang_left
                        force_ang_right = np.arcsin(norm_right[1])
                        if norm_right[0] < 0:
                            force_ang_right = np.pi - force_ang_right
                        while force_ang_right < 0 or force_ang_right > 2 * np.pi or force_ang_left < 0 or force_ang_left > 2 * np.pi:
                            if force_ang_right < 0:
                                force_ang_right += 2 * np.pi  # prevents angles from blowing up
                            if force_ang_left < 0:
                                force_ang_left += 2 * np.pi
                            if force_ang_right > 2 * np.pi:
                                force_ang_right -= 2 * np.pi
                            if force_ang_left > 2 * np.pi:
                                force_ang_left -= 2 * np.pi
                        if force_ang_right - force_ang_left < np.pi:
                            force_ang = 0.5 * (
                                        force_ang_right + force_ang_left)  # determines the direction the forces point in
                        else:
                            force_ang = 0.5 * (force_ang_right + force_ang_left) - np.pi
                    forces_str.append(
                        Force(vertex[0], vertex[1], np.linalg.norm(force_sum), force_ang))  # adds them to a list

                # calculate forces
                for i in range(args.n_beads):
                    bead = beads[args.n_beads - i]
                    contact_forces = []
                    if i == 0:
                        right_force_1 = tension  # forces for rightmost bead
                        right_force_2 = F_app
                        str_force_r = Force(beads[args.n_beads].pos[0], beads[args.n_beads].pos[1], 0,
                                            np.pi - beads[args.n_beads].theta)
                        right_force_3 = Force(args.w * (args.n_beads - 1), 0, 0, np.pi / 2)
                    else:
                        right_force_1 = copy.copy(left_force_1)  # forces from bead to the right
                        right_force_2 = copy.copy(left_force_2)
                        right_force_3 = Force(args.w * (args.n_beads - 1), 0, 0, np.pi / 2)
                        str_force_r = forces_str[2 * args.n_beads - 2 * i]
                        if right_force_1.ang > np.pi:
                            right_force_1.ang -= np.pi  # opposite direction
                        else:
                            right_force_1.ang += np.pi
                        if right_force_2.ang > np.pi:
                            right_force_2.ang -= np.pi
                        else:
                            right_force_2.ang += np.pi
                    str_force_l = forces_str[2 * args.n_beads - 1 - 2 * i]
                    gravity = Force(0.5 * (bead.pos[0] + bead.C[0]), 0.5 * (bead.pos[1] + bead.C[1]),
                                    args.grav_const * 0.0031, 3 * np.pi / 2)
                    contact_forces.append(right_force_1)  # all the non-reaction forces on a bead
                    contact_forces.append(right_force_2)
                    contact_forces.append(right_force_3)
                    contact_forces.append(str_force_r)
                    contact_forces.append(str_force_l)
                    contact_forces.append(gravity)

                    # types of contact
                    if contact_type[args.n_beads - i - 1] == 'two':
                        # 2 contact points
                        if disp_ini[args.n_beads - i - 1] >= 0:  # contact point is different if disp is negative
                            force_top = Force(beads[args.n_beads - i].A[0], beads[args.n_beads - i].A[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu)+args.mis)
                            force_bottom = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu)-args.mis)
                        else:
                            force_top = Force(beads[args.n_beads - i].A[0], beads[args.n_beads - i].A[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu)+args.mis)
                            force_bottom = Force(beads[args.n_beads - i].G[0], beads[args.n_beads - i].G[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 - np.arctan(args.mu))
                        contact_forces.append(force_top)
                        contact_forces.append(force_bottom)
                        x, y = fsolve(two_point_contact, (1, 1), contact_forces)  # solves for the force magnitudes
                        if x <= 0: x = 0  # cant be negative
                        if y <= 0: y = 0
                        force_top.mag = x
                        force_bottom.mag = y
                        for force in contact_forces:  # force sum
                            moment[args.n_beads - i - 1] += -force.mag * np.cos(force.ang) * (
                                        force.ypos) + force.mag * (force.xpos) * np.sin(
                                force.ang)
                            y_sum[args.n_beads - i - 1] += force.mag * np.sin(force.ang)

                        left_force_1 = force_top
                        left_force_2 = force_bottom

                    elif contact_type[args.n_beads - 1 - i] == 'one':
                        # one contact point
                        args.mu=0.2
                        if disp_ini[args.n_beads - 1 - i] >= 0:
                            forces_norm = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                args.thtb - np.pi / 2 - beads[
                                                    args.n_beads - i].theta-args.mis)  # normal, frictional forces
                            forces_fric = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                args.thtb - beads[args.n_beads - i].theta-args.mis)
                        else:
                            forces_norm = Force(beads[args.n_beads - i].G[0], beads[args.n_beads - i].G[1], 0,
                                                args.thtb - np.pi / 2 - beads[args.n_beads - i - 1].theta)
                            forces_fric = Force(beads[args.n_beads - i].G[0], beads[args.n_beads - i].G[1], 0,
                                                args.thtb - beads[args.n_beads - i - 1].theta)
                        contact_forces.append(forces_norm)
                        contact_forces.append(forces_fric)
                        x, y = fsolve(point_contact, (1, 1), contact_forces)  # solves for magnitudes
                        if y > x * args.mu:
                            y = x * args.mu
                        elif y < -x * args.mu:
                            y = -x * args.mu
                        forces_norm.mag = x
                        forces_fric.mag = y
                        # need an exception for loop caused by separation of second bead
                        for force in contact_forces:  # force sum
                            moment[args.n_beads - i - 1] += -force.mag * np.cos(force.ang) * (
                                        force.ypos - bead.G[1]) + force.mag * (force.xpos - bead.G[0]) * np.sin(
                                force.ang)
                            y_sum[args.n_beads - i - 1] += force.mag * np.sin(force.ang)
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
                    all_forces.append(gravity)
                    all_forces.append(right_force_3)
                for s in range(args.n_beads):
                    if np.abs(moment[s]) < 1e-9:
                        moment[s] = 0
                    if np.abs(y_sum[s]) < 1e-9:
                        y_sum[s] = 0

                # rewrite initial string contact
                # calculate angle at string contact
                # add moment, ysum/10 (bounded by crit angle)

                for s in range(args.n_beads):
                    for t in range(s + 1):
                        tht_ini[args.n_beads - t - 1] -= max(min(0.05, moment[args.n_beads - s - 1]/100),
                                                             -0.05)  # adjust angle
                        if tht_ini[args.n_beads - t - 1] < tht_ini[args.n_beads - s - 1]:
                            tht_ini[args.n_beads - t - 1] = tht_ini[args.n_beads - s - 1]
                for s in range(args.n_beads):
                    if contact_type[args.n_beads - s - 1] == 'one':

                        disp_ini[args.n_beads - s - 1] -= max(min(0.0005, y_sum[args.n_beads - s - 1] / 20000),
                                                          -0.0005)

                        tht_ini[args.n_beads - t - 1] -= max(min(0.005, moment[args.n_beads - s - 1] / 300),
                                                             -0.005)

                    elif contact_type[args.n_beads - s - 1] == 'two':
                        if all_forces[4+len(contact_forces)*s].mag == 0:
                            contact_type[args.n_beads - s - 1] = 'one'
                        if s == args.n_beads - 1:
                            disp_ini[args.n_beads - s - 1] = max_disp(tht_ini[args.n_beads - s - 1], args)
                        else:
                            disp_ini[args.n_beads - s - 1] = max_disp(
                                tht_ini[args.n_beads - s - 1] - tht_ini[args.n_beads - s - 2], args)
                    else:
                        if y_sum[args.n_beads - s - 1] != 0:
                            contact_type[args.n_beads - s - 1] = 'two'
                            disp_ini[args.n_beads - s - 1] -= max(min(0.0005, y_sum[args.n_beads - s - 1] / 20000),
                                                                  -0.0005)
                            if s == args.n_beads - 1:
                                tht_ini[args.n_beads - s - 1] = min_ang(disp_ini[args.n_beads - s - 1], args)
                            else:
                                tht_ini[args.n_beads - s - 1] = min_ang(disp_ini[args.n_beads - s - 1], args) + \
                                                                tht_ini[args.n_beads - s - 2]
                            for t in range(s + 1):
                                tht_ini[args.n_beads - s - 1 + t] = min_ang(disp_ini[args.n_beads - s - 1 + t],
                                                                            args) + tht_ini[
                                                                        args.n_beads - s + t - 2]

                    '''''''''''''''''''''''''''''''''''''''''''''''''''
                    if s == args.n_beads - 1:  # all this to check if rotation will create one or 2 contacts
                        out_ang = args.thtb
                    else:
                        out_ang = -tht_ini[args.n_beads - s - 2] + args.thtb
                    across_ang = np.pi - args.thtb - tht_ini[args.n_beads - s - 1]

                    if out_ang + np.pi / 2 < across_ang:
                        contact_type[args.n_beads - s - 1] = 'two'
                        if s == args.n_beads - 1:
                            disp_ini[args.n_beads - s - 1] = max_disp(tht_ini[args.n_beads - s - 1], args)
                        else:
                            disp_ini[args.n_beads - s - 1] = max_disp(
                                tht_ini[args.n_beads - s - 1] - tht_ini[args.n_beads - s - 2], args)

                    else:
                        contact_type[args.n_beads - s - 1] = 'one'
                        disp_ini[args.n_beads - s - 1] -= max(min(0.0005, y_sum[args.n_beads - s - 1] / 20000),
                                                              -0.0005)
                        if ((s==args.n_beads-1 and disp_ini[args.n_beads-s-1]>=max_disp(tht_ini[args.n_beads-s-1],args)) or disp_ini[args.n_beads-s-1]>max_disp(tht_ini[args.n_beads-s-1]-tht_ini[args.n_beads-s-2],args)) and y_sum[args.n_beads-s-1]<0:
                            if s == args.n_beads - 1:
                                disp_ini[args.n_beads - s - 1] = max_disp(tht_ini[args.n_beads - s - 1], args)
                            else:
                                disp_ini[args.n_beads - s - 1] = max_disp(
                                    tht_ini[args.n_beads - s - 1] - tht_ini[args.n_beads - s - 2], args)
                            contact_type[args.n_beads-s-1]='two'
                    '''''''''''''''''''''''''''''''''''''''
                    if s == args.n_beads - 1:
                        if tht_ini[0] <= 0:
                            contact_type[0] = 'two'
                    else:
                        if tht_ini[args.n_beads - s - 1] <= tht_ini[args.n_beads - s - 2]:
                            contact_type[args.n_beads - s - 1] = 'two'
                            tht_ini[args.n_beads - s - 1] = tht_ini[args.n_beads - s - 2]
                            disp_ini[args.n_beads - s - 1] = 0
                for s in range(args.n_beads - 1):
                    if tht_ini[s] >= tht_ini[s + 1]:
                        tht_ini[s + 1] = tht_ini[s]
                        contact_type[s + 1] = "two"


                progress += 1
                if progress >= 1000:  # no infinite loops
                    print(moment)
                    print(y_sum)
                    print(tht_ini)
                    print(disp_ini)
                    print(m)
                    print(j)
                    break
                #debugging checks
                if m == 4:
                    m = 4
                if m == 78:
                    m = 78
                if tht_ini[args.n_beads - 1] >= 1:  # beyond this, system does not make physical sense
                    print('maxxed out')
                    print(m)
                    finit = True
                    break
                prev_val_tht = tht_ini[0]
                prev_val_disp = disp_ini[0]

            if m % 10 == 10:

                shapes = []  # for making the pictures
                # for this bead and all beads to the right
                # add displacement, angle around the contact point
                for bead in beads:
                    top = [bead.A, bead.B, bead.D, bead.C]
                    bot = [bead.E, bead.F, bead.H, bead.G]
                    top_poly = plt.Polygon(top)
                    bot_poly = plt.Polygon(bot)
                    shapes.append(top_poly)
                    shapes.append(bot_poly)  # makes a list of shapes
                plt.axes()

                for shape in shapes:
                    plt.gca().add_patch(shape)  # plots shapes
                    count = -1
                for force in all_forces:  # plots arrows as forces
                    count += 1
                    if force.mag <= 1e-6:
                        continue
                    if np.floor(count / 8) % 3 == 0:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 3000,
                                        force.mag * np.sin(force.ang) / 3000, color='black', width=0.0005)
                    elif np.floor(count / 8) % 3 == 1:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 3000,
                                        force.mag * np.sin(force.ang) / 3000, color='orange', width=0.0005)
                    else:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 3000,
                                        force.mag * np.sin(force.ang) / 3000, color='green', width=0.0005)
                    plt.gca().add_patch(bow)
                plt.axis('scaled')
                plt.plot(string_arr[:, 0], string_arr[:, 1], color='red')
                plt.show()
            exte = (args.R - beads[args.n_beads].B[1])  # plots the vertical displacement of the top right corner of the rightmost bead
            extension[m, j] = exte * 800
            for i in range(args.n_beads):
                fin_disp[m, i, j] = disp_ini[i]  # collects data for graphs
                fin_tht[m, i, j] = tht_ini[i]
    force = np.linspace(0, F, args.n_iters)
    # for i in range(args.param_sweep):                    #displacement plot
    # plt.plot(force,fin_disp[:,args.n_beads-1,i])
    # plt.legend(['0.011','0.013','0.015','0.017','0.019'])
    # plt.xlabel('Force [N]')
    # plt.ylabel('displacement [m]')
    # plt.show()
    # for i in range(args.param_sweep):                    #angle plot
    # plt.plot(force,fin_tht[:,args.n_beads-1,i])
    # plt.legend(['0.011','0.013','0.015','0.017','0.019'])
    # plt.xlabel('Force [N]')
    # plt.ylabel('Angular displacement [radians]')
    # plt.show()

    test_one = pandas.read_csv(r"One_bead_1").to_numpy()
    test_two = pandas.read_csv(r"One_bead_2").to_numpy()
    test_thre = pandas.read_csv(r"One_bead_3").to_numpy()
    one_fo = test_one[3:, 2].astype('float64')
    one_di = test_one[3:, 1].astype('float64')

    two_fo = test_two[3:, 2].astype('float64')
    two_di = test_two[3:, 1].astype('float64')

    thre_fo = test_thre[3:, 2].astype('float64')
    thre_di = test_thre[3:, 1].astype('float64')
    for ex in range(len(thre_di)):
        thre_di[ex] -= 2
    # plt.plot(one_di, one_fo)
    # plt.plot(two_di, two_fo)
    # plt.plot(thre_di, thre_fo)
    twenty_N = pandas.read_csv(r"30_deg_fancy").to_numpy()
    #plt.plot(twenty_N[:,0],twenty_N[:,1])
    #plt.plot(twenty_N[:,3],twenty_N[:,4])
    #plt.plot(twenty_N[:, 6], twenty_N[:, 7])
    #plt.plot(twenty_N[:, 9], twenty_N[:, 10])
    seventy=pandas.read_csv(r"70deg_10_50_90_120N").to_numpy()
    fifty=pandas.read_csv(r"50deg_10_50_90_120N").to_numpy()
    forty=pandas.read_csv(r"40deg_10_50_90_120N").to_numpy()
    thirty=pandas.read_csv(r"30deg_10_50_90_120N").to_numpy()
    #plt.plot(seventy[:, 0], seventy[:, 1])
    #plt.plot(fifty[:,0],fifty[:,1])
    #plt.plot(fifty[:,2],fifty[:,3])
    #plt.plot(fifty[:,4],fifty[:,5])
    plt.plot(forty[:,0],forty[:,1])
    plt.plot(forty[:,2],forty[:,3])
    plt.plot(forty[:,4],forty[:,5])
    #plt.plot(thirty[:,0],thirty[:,1])
    #plt.plot(thirty[:,2],thirty[:,3])
    #plt.plot(thirty[:,4],thirty[:,5])
    read_data = pandas.read_csv(r"one_bead_test").to_numpy()
    read_ext = read_data[2:, 1].astype('float64')
    read_for = read_data[2:, 2].astype('float64')
    # plt.plot(read_ext, read_for)
    '''''''''''''''''''''
    for p in range(args.param_sweep):            #extension plot
        for i in range(m):
            if extension[i,p]!=0 and extension[i-1,p]==0:
                for j in range(i):
                    extension[j,p]=extension[i,p]*j/i
    '''''''''''''''''''''
    adj_ext=np.zeros(m)
    for j in range(args.param_sweep):
        for i in range(m):
            adj_ext[i]=extension[i,j]-extension[0,j]
    for i in range(args.param_sweep):  # extension plot
        plt.plot(extension[:, i]-extension[0,i], (force))
    plt.ylabel('Force [N]')
    plt.xlabel('Vertical displacement [mm]')
    plt.xlim([-0.1, 25])
    plt.legend(['Experimental data', 'Model data'])
    steel_test = pandas.read_csv(r"Steel_cable_resin_beads.csv").to_numpy()
    fo = steel_test[1:2450, 2].astype('float64')
    di = steel_test[1:2450, 1].astype('float64')
    fo *= (-1)
    di *= (-1)
    # plt.plot(di, fo)

    plt.show()
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
    plt.show()
    f = r"c://Users/Chris/Desktop/animation.mp4"
    plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Chris\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'

    # ani.save(f,writer='ffmpeg')


main()
