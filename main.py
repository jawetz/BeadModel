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
from contacts import surface_contact,point_contact, two_point_contact,positioning,friction
from no_overlap import max_disp, min_ang
from string_crt_pt import two_pt_crt_pt
import argparse

def init():
    shapes.set_data([],[])
    return shapes
def animate(i):

    shapes = []  # for making the pictures
    # for this bead and all beads to the right
    # add displacement, angle around the contact point
    beads[0] = Bead([0, 0], 0, 0, args.w, args.R, args.r, args.thtb)
    for j in range(args.n_beads):
        new_pos = bead_pos(beads[j], fin_tht[i,j,0], fin_disp[i,j,0])  # adds beads to the system
        new_bead = Bead(new_pos, fin_tht[i,j,0], fin_disp[i,j,0], args.w, args.R, args.r, args.thtb)
        beads[j+1] = new_bead
    for bead in beads:
        top = [bead.A, bead.B, bead.D, bead.C]
        bot = [bead.E, bead.F, bead.H, bead.G]
        top_poly = plt.Polygon(top)
        bot_poly = plt.Polygon(bot)
        shapes.append(top_poly)
        shapes.append(bot_poly)  # makes a list of shapes

    for shape in shapes:
        plt.gca().add_patch(shape)  # plots shapes
    plt.xlim([-0.05,0.1])
    plt.ylim([-0.06,0.03])
    return shapes

parser=argparse.ArgumentParser(description='Bead Model')
parser.add_argument('--n_iters',type=int,default=100,help='granularity of simulation')
parser.add_argument('--param_sweep',type=int,default=7,help='for varying parameters across trials')
parser.add_argument('--n_beads',type=int,default=6,help='number of non-fixed beads in the system')
parser.add_argument('--R',type=float,default=0.0075,help='outer radius')
parser.add_argument('--r',type=float,default=0.0002,help='inner bead radius')
parser.add_argument('--w',type=float,default=0.015,help='bead length')
parser.add_argument('--ys',type=float,default=145.58,help='youngs modulus times cross-sectional area')
parser.add_argument('--delt',type=float,default=0.001,help='initial displacement of string')
parser.add_argument('--thtb',type=float,default=np.pi/6,help='bead angle')
parser.add_argument('--mu',type=float,default=0.2,help='friction coefficient')
parser.add_argument('--grav_const',type=float,default=9.81,help='gravitational constant')
parser.add_argument('--dens',type=float,default=1000,help='density of the material')
args=parser.parse_args()

ys=145.58       #youngs modulus*area
tot_len=(args.n_beads+1)*args.w+0.29  #total length of the string (including the part behind the beads)
k=ys/tot_len
print(k)
d = (args.R) / np.sin(args.thtb)  # length of angled side

beads = np.zeros(args.n_beads + 1, dtype=Bead)
strings = np.zeros((2 * args.n_beads + 2, 2))
str_type = np.empty(2 * args.n_beads + 2,dtype=np.dtype('U100'))
all_forces = []
shapes = []
tht = np.linspace(0.001, 1, args.n_iters)

fin_tht = np.zeros((args.n_iters, args.n_beads,args.param_sweep))
fin_disp = np.zeros((args.n_iters, args.n_beads,args.param_sweep))
extension=np.zeros((args.n_iters, args.param_sweep))



def main():
    # initialize beads
    for j in range(args.param_sweep):
        contact_type = np.empty(args.n_beads, dtype=np.dtype('U100'))
        args.mu=0.2+0.05*j
        d = (args.R) / np.sin(args.thtb)  # length of angled side

        # tht_ini_ini = np.random.random_sample(args.n_beads) / args.n_beads
        tht_ini = [0.0]*args.n_beads
        # disp_ini = np.random.random_sample(args.n_beads) * 0.001
        disp_ini = [0.00]*args.n_beads
        for i in range(args.n_beads):
            contact_type[args.n_beads - 1 - i] = 'surface'
        finit=False
        for m in range(args.n_iters):
            global F
            F=m/args.n_iters            #applied force
            #initial bead interface
            # disp_ini = np.random.random_sample(args.n_beads) * 0.001
            beads[0] = Bead([0, 0], 0, 0, args.w, args.R, args.r, args.thtb)  #fixed bead
            moment=np.ones(args.n_beads)
            y_sum=np.ones(args.n_beads)
            progress=0
            while (np.amax(np.abs(moment)) > 0.0001 or np.amax(np.abs(y_sum)) >0.00005) and finit==False:         #loops until convergence
                moment = np.zeros(args.n_beads)
                y_sum = np.zeros(args.n_beads)
                for i in range(args.n_beads):
                    # must be difference of tht_ini values when more than one bead
                    if max_disp(tht_ini[i],args) < disp_ini[i]:
                        if i==0:
                            disp_ini[i] = max_disp(tht_ini[i],args)      #ensures initial beads dont overlap
                        else:
                            disp_ini[i] = max_disp(tht_ini[i]-tht_ini[i-1],args)

                for i in range(1, args.n_beads + 1):
                    new_pos = bead_pos(beads[i - 1], tht_ini[i - 1], disp_ini[i - 1])           #adds beads to the system
                    new_bead = Bead(new_pos, tht_ini[i - 1], disp_ini[i - 1], args.w, args.R, args.r, args.thtb)
                    beads[i] = new_bead

                # initialize string points
                strings[0, :] = beads[0].pos[0] - args.w, beads[0].pos[1]      #initializes string
                str_type[0] = 'start'
                #need to fix how string len is calculated, maybe more accurate no-contact positioning?
                for i in range(2 * args.n_beads):
                    if i % 2 == 0:
                        out = False         #whether string is going in or out of a bead
                    else:
                        out = True
                    bead_no = np.floor(i / 2).astype(int)

                    st_pt = strings[i, :]
                    result = intersect(bead_no, st_pt, out,beads)
                    if out:
                        if result == 'top':             #adds string points by where they contact the bead
                            strings[i + 1, :] = beads[bead_no + 1].C
                            str_type[i + 1] = 'top'
                        elif result == 'bot':
                            strings[i + 1, :] = beads[bead_no + 1].E
                            str_type[i + 1] = 'bot'
                        else:
                            strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                            str_type[i + 1] = 'none'

                    else:
                        if result == 'top':
                            strings[i + 1, :] = beads[bead_no].D
                            str_type[i + 1] = 'top'
                        elif result == 'bot':
                            strings[i + 1, :] = beads[bead_no].F
                            str_type[i + 1] = 'bot'
                        else:
                            strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                            str_type[i + 1] = 'none'
                        if np.floor(i/2).astype(int) == 0:
                            if tht_ini[np.floor(i/2).astype(int)]==0:
                                strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                                str_type[i + 1] = 'none'
                        else:
                            if tht_ini[np.floor(i/2).astype(int)]==tht_ini[np.floor(i/2).astype(int)-1]:
                                strings[i + 1, :] = [beads[bead_no].pos[0], beads[bead_no].pos[1]]
                                str_type[i + 1] = 'none'
                strings[2 * args.n_beads + 1, :] = beads[args.n_beads].pos
                str_type[2*args.n_beads+1] = 'end'
                for s in range(len(str_type)):
                    if str_type[s]=='none' and str_type[s-1] != 'none':
                        left_pt=strings[s-1,:]
                        t=0
                        for t in range(s+1,len(str_type)):
                            if str_type[t] != 'none':
                                break
                        right_pt=strings[t,:]
                        for u in range(s,t):
                            coeff=(u-s+1)/(t-s)
                            strings[u]=((u-s+1)/(t-s+1))*(right_pt-left_pt)+(left_pt)
                global string_arr
                string_arr = np.vstack(strings)
                # calculate string length
                string_len = (-args.w * (args.n_beads+1)) + args.delt        #initial string length
                for i in range(2 * args.n_beads + 1):
                    dist = np.sqrt(
                        (string_arr[i, 0] - string_arr[i + 1, 0]) ** 2 + (string_arr[i, 1] - string_arr[i + 1, 1]) ** 2)
                    string_len += dist                          #stretched string length
                tension = Force(beads[args.n_beads].pos[0], beads[args.n_beads].pos[1], k * string_len, np.pi - beads[args.n_beads].theta)
                F_app = Force(args.w*(args.n_beads),0,
                              F, 3 * np.pi / 2)                 #adds tension, applied force
                # calculate string forces
                forces_str = []
                #need to come up with exceptions for no contact case
                for i in range(1, 2 * args.n_beads + 1):             #calculates string-bead contact
                    flag='contact'
                    vertex = string_arr[i, :]
                    left = string_arr[i - 1, :]
                    right = string_arr[i + 1, :]
                    left_vec = left - vertex
                    right_vec = right - vertex
                    if np.linalg.norm(left_vec) == 0:
                        flag='no contact'
                    else:
                        norm_left = left_vec / np.linalg.norm(left_vec)         #points in the two directions
                    if np.linalg.norm(right_vec) == 0:
                        flag='no contact'
                    else:
                        norm_right = right_vec / np.linalg.norm(right_vec)
                    if flag=='no contact':
                        force_sum = [0,0]
                        force_ang = 10
                    else:
                        force_sum = norm_left * tension.mag + norm_right * tension.mag  #vector sum of tension forces
                        force_ang_left = np.arcsin(norm_left[1])
                        if norm_left[0] < 0:
                            force_ang_left = np.pi - force_ang_left
                        force_ang_right = np.arcsin(norm_right[1])
                        if norm_right[0] < 0:
                            force_ang_right = np.pi - force_ang_right
                        while force_ang_right < 0 or force_ang_right > 2 * np.pi or force_ang_left < 0 or force_ang_left > 2 * np.pi:
                            if force_ang_right < 0:
                                force_ang_right += 2 * np.pi            #prevents angles from blowing up
                            if force_ang_left < 0:
                                force_ang_left += 2 * np.pi
                            if force_ang_right > 2 * np.pi:
                                force_ang_right -= 2 * np.pi
                            if force_ang_left > 2 * np.pi:
                                force_ang_left -= 2 * np.pi
                        if force_ang_right - force_ang_left < np.pi:
                            force_ang = 0.5 * (force_ang_right + force_ang_left)            #determines the direction the forces point in
                        else:
                            force_ang = 0.5 * (force_ang_right + force_ang_left) - np.pi
                    forces_str.append(Force(vertex[0], vertex[1], np.linalg.norm(force_sum), force_ang))        #adds them to a list

                # calculate forces
                for i in range(args.n_beads):
                    bead = beads[args.n_beads - i]
                    contact_forces = []
                    if i == 0:
                        right_force_1 = tension         #forces for rightmost bead
                        right_force_2 = F_app
                        str_force_r = Force(beads[args.n_beads].pos[0], beads[args.n_beads].pos[1], 0, np.pi - beads[args.n_beads].theta)
                        right_force_3= Force(args.w*(args.n_beads-1),0,0, np.pi / 2)
                    else:
                        right_force_1 = copy.copy(left_force_1)         #forces from bead to the right
                        right_force_2 = copy.copy(left_force_2)
                        right_force_3= Force(args.w*(args.n_beads-1),0,0, np.pi / 2)
                        str_force_r = forces_str[2 * args.n_beads - 2 * i]
                        if right_force_1.ang > np.pi:
                            right_force_1.ang -= np.pi                  #opposite direction
                        else:
                            right_force_1.ang += np.pi
                        if right_force_2.ang > np.pi:
                            right_force_2.ang -= np.pi
                        else:
                            right_force_2.ang += np.pi
                    str_force_l = forces_str[2 * args.n_beads - 1 - 2 * i]
                    gravity=Force(0.5*(bead.pos[0]+bead.C[0]),0.5*(bead.pos[1]+bead.C[1]),args.grav_const*bead.vol*args.dens,3*np.pi/2)
                    contact_forces.append(right_force_1)            #all the non-reaction forces on a bead
                    contact_forces.append(right_force_2)
                    contact_forces.append(right_force_3)
                    contact_forces.append(str_force_r)
                    contact_forces.append(str_force_l)
                    contact_forces.append(gravity)

                    # types of contact
                    if contact_type[args.n_beads - i - 1] == 'surface':
                        # contact surface
                        surface_top = Force(bead.A[0], bead.A[1],
                                            0, np.pi / 2 - args.thtb-bead.theta)  # equivalent surface force
                        if args.thtb<np.pi/4:
                            surface_bot = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                            args.thtb - np.pi / 2-bead.theta)  # equivalent surface force bottom
                        else:
                            surface_bot = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                                args.thtb - np.pi / 2 - bead.theta)  # equivalent surface force bottom
                        contact_forces.append(surface_top)
                        contact_forces.append(surface_bot)
                        x, y = fsolve(surface_contact, (1, 1), contact_forces)        #solves for force magnitudes

                        mome=0
                        for force in contact_forces[0:6]:
                            mome += -force.mag * np.cos(force.ang) * (force.ypos-bead.G[1]) + force.mag * (force.xpos-bead.G[0]) * np.sin(force.ang)
                        surface_top.mag=x
                        surface_bot.mag=y
                        q=fsolve(positioning,(.001),(mome,surface_top,surface_bot,bead))
                        q=q[0]
                        if q>d:
                            surface_top.xpos = beads[args.n_beads - i - 1].pos[0]
                            surface_top.ypos = beads[args.n_beads - i - 1].pos[1]
                            surface_bot.xpos = bead.G[0]
                            surface_bot.ypos = bead.G[1]
                            surface_top.mag=0
                            surface_bot.mag=0
                            s, t, u = fsolve(friction, (x, y, 0.1), (contact_forces, bead, args))
                            surface_top.mag = s
                            surface_bot.mag = t
                            if u > np.arctan(args.mu):
                                u = np.arctan(args.mu)
                            surface_top.ang += u
                            if args.thtb >= np.pi / 4:
                                surface_bot.ang += u
                            else:
                                surface_bot.ang -= u
                        else:
                            surface_top.xpos+=q*np.cos(args.thtb+bead.theta)
                            surface_top.ypos-=q*np.sin(args.thtb+bead.theta)
                            surface_bot.xpos-=q*np.cos(args.thtb-bead.theta)
                            surface_bot.ypos-=q*np.sin(args.thtb-bead.theta)
                        for force in contact_forces:            #force sums
                            moment[args.n_beads-i-1] += -force.mag * np.cos(force.ang) * (force.ypos-bead.G[1]) + force.mag * (force.xpos-bead.G[0]) * np.sin(
                                force.ang)
                            y_sum[args.n_beads-i-1] += force.mag * np.sin(force.ang)

                        left_force_1 = surface_top        #prepares the next bead
                        left_force_2 = surface_bot
                    elif contact_type[args.n_beads - i - 1] == 'two':
                        # 2 contact points
                        if disp_ini[args.n_beads - i - 1] >= 0:                                  #contact point is different if disp is negative
                            force_top = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu))
                            force_bottom = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 - np.arctan(args.mu))
                        else:
                            force_top = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu))
                            force_bottom = Force(beads[args.n_beads - i].G[0], beads[args.n_beads - i].G[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 - np.arctan(args.mu))
                        contact_forces.append(force_top)
                        contact_forces.append(force_bottom)
                        x, y = fsolve(two_point_contact, (1, 1), contact_forces)            #solves for the force magnitudes
                        if x <= 0: x = 0                                                    #cant be negative
                        if y <= 0: y = 0
                        force_top.mag = x
                        force_bottom.mag = y
                        for force in contact_forces:                #force sum
                            moment[args.n_beads-i-1] += -force.mag * np.cos(force.ang) * (force.ypos-bead.G[1]) + force.mag * (force.xpos-bead.G[0]) * np.sin(
                                force.ang)
                            y_sum[args.n_beads-i-1] += force.mag * np.sin(force.ang)

                        left_force_1 = force_top
                        left_force_2 = force_bottom

                    elif contact_type[args.n_beads-1-i]=='one':
                        # one contact point
                        if disp_ini[args.n_beads-1-i] >= 0:
                            forces_norm = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                args.thtb - np.pi / 2 - beads[args.n_beads - i].theta)                #normal, frictional forces
                            forces_fric = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                args.thtb - beads[args.n_beads - i].theta)
                        else:
                            forces_norm = Force(beads[args.n_beads - i].G[0], beads[args.n_beads-i].G[1], 0,
                                                args.thtb - np.pi / 2 - beads[args.n_beads - i - 1].theta)
                            forces_fric = Force(beads[args.n_beads - i].G[0], beads[args.n_beads-i].G[1], 0,
                                                args.thtb - beads[args.n_beads - i - 1].theta)
                        contact_forces.append(forces_norm)
                        contact_forces.append(forces_fric)
                        x, y = fsolve(point_contact, (1, 1), contact_forces)        #solves for magnitudes
                        if y > x * args.mu:
                            y = x * args.mu
                        elif y < -x*args.mu:
                            y=-x*args.mu
                        forces_norm.mag = x
                        forces_fric.mag = y
                        #need an exception for loop caused by separation of second bead
                        for force in contact_forces:            #force sum
                            moment[args.n_beads-i-1] += -force.mag * np.cos(force.ang) * (force.ypos-bead.G[1]) + force.mag * (force.xpos-bead.G[0]) * np.sin(
                                force.ang)
                            y_sum[args.n_beads-i-1] += force.mag * np.sin(force.ang)
                        left_force_1 = forces_norm
                        left_force_2 = forces_fric
                    else:
                        if disp_ini[args.n_beads - i - 1] >= 0:                                  #contact point is different if disp is negative
                            force_top = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu))
                            force_bottom = Force(beads[args.n_beads - i - 1].H[0], beads[args.n_beads - i - 1].H[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 - np.arctan(args.mu))
                        else:
                            force_top = Force(beads[args.n_beads - i - 1].pos[0], beads[args.n_beads - i - 1].pos[1], 0,
                                              np.pi - args.thtb - bead.theta - np.pi / 2 + np.arctan(args.mu))
                            force_bottom = Force(beads[args.n_beads - i].G[0], beads[args.n_beads - i].G[1], 0,
                                                 args.thtb - bead.theta - np.pi / 2 - np.arctan(args.mu))
                        contact_forces.append(force_top)
                        contact_forces.append(force_bottom)
                        x, y = fsolve(two_point_contact, (1, 1), contact_forces)            #solves for the force magnitudes
                        if x <= 0: x = 0                                                    #cant be negative
                        if y <= 0: y = 0
                        force_top.mag = x
                        force_bottom.mag = y
                        for force in contact_forces:                #force sum
                            moment[args.n_beads-i-1] += -force.mag * np.cos(force.ang) * (force.ypos-bead.G[1]) + force.mag * (force.xpos-bead.G[0]) * np.sin(
                                force.ang)
                            y_sum[args.n_beads-i-1] += force.mag * np.sin(force.ang)

                        left_force_1 = force_top
                        left_force_2 = force_bottom
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
                    if np.abs(moment[s])<1e-9:
                        moment[s]=0
                    if np.abs(y_sum[s])<1e-9:
                        y_sum[s]=0
                if str_type[1]=='bot' and str_type[2]!='top' and contact_type[0]=='two' and args.thtb<=np.pi/4:
                    moment[0]=moment[0]/10
                for s in range(args.n_beads):
                    for t in range(s+1):
                        tht_ini[args.n_beads - t - 1] -= max(min(0.05, moment[args.n_beads-s-1]*5), -0.05)  # adjust angle
                skip=False
                contact_typ = 'other'
                for s in range(args.n_beads):
                    if s==0:
                        if tht_ini[0] < 0 or contact_type[0]=='other':
                            tht_ini[0]+=max(min(0.05, moment[0]*10), -0.05)
                            disp_ini[0] -= max(min(0.0005, y_sum[0] / 2000),-0.0005)
                            contact_type[0]='other'
                            skip=True
                            if np.amax(np.abs(moment)) < 0.0001 and np.amax(np.abs(y_sum)) <0.00005:
                                contact_type[0]='one'
                                tht_ini[0] = min_ang(disp_ini[0], args)
                            elif np.amax(np.abs(y_sum)) < 0.00005 and np.amax(moment)>=0.0001:
                                disp_ini[0]-=max(min(0.0001,moment[0]/100),-0.0001)
                                tht_ini[0] = min_ang(disp_ini[0], args)
                            elif np.amax(np.abs(y_sum)) < 0.00005 and np.amax(np.abs(moment))>=0.0001:
                                tht_ini[0]-=moment[0]*10
                                contact_typ='one'
                            else:
                                tht_ini[0] = min_ang(disp_ini[0], args)

                    else:
                        if tht_ini[s]<tht_ini[s-1] or contact_type[0]=='other':
                            tht_ini[s]=tht_ini[s-1]
                if contact_typ=='one':
                    contact_type[0]='one'
                for s in range(args.n_beads):
                    if s == args.n_beads - 1:  # all this to check if rotation will create one or 2 contacts
                        out_ang = args.thtb
                    else:
                        out_ang = -tht_ini[args.n_beads - s - 2] + args.thtb
                    across_ang = np.pi - args.thtb - tht_ini[args.n_beads - s - 1]

                    if out_ang+np.pi/2<across_ang:
                        contact_type[args.n_beads-s-1]='two'
                        if s==args.n_beads-1:
                            disp_ini[args.n_beads-s-1]=max_disp(tht_ini[args.n_beads-s-1],args)
                        else:
                            disp_ini[args.n_beads-s-1]=max_disp(tht_ini[args.n_beads-s-1]-tht_ini[args.n_beads-s-2],args)

                    elif skip==False:
                        contact_type[args.n_beads-s-1]='one'
                        disp_ini[args.n_beads - s - 1] -= max(min(0.0005, y_sum[args.n_beads-s-1] / 2000), -0.0005)
                    if s==args.n_beads-1:
                        if tht_ini[0] <= 0:
                            contact_type[0]='surface'
                    else:
                        if tht_ini[args.n_beads-s-1]<=tht_ini[args.n_beads-s-2]:
                            contact_type[args.n_beads-s-1]='surface'

                progress+=1
                if progress>=1000:          #no infinite loops
                    print(moment)
                    print(tht_ini)
                    print(m)
                    print(j)
                    break
                if m==40:
                    m=40
                if m==88:
                    m=88
                if tht_ini[0]>=1.57:            #beyond this, system does not make physical sense
                    print('maxxed out')
                    print(m)
                    finit=True
                    break


            if m % 20 == 0:

                shapes = []             #for making the pictures
                # for this bead and all beads to the right
                # add displacement, angle around the contact point
                for bead in beads:
                    top = [bead.A, bead.B, bead.D, bead.C]
                    bot = [bead.E, bead.F, bead.H, bead.G]
                    top_poly = plt.Polygon(top)
                    bot_poly = plt.Polygon(bot)
                    shapes.append(top_poly)
                    shapes.append(bot_poly)             #makes a list of shapes
                plt.axes()

                for shape in shapes:
                    plt.gca().add_patch(shape)          #plots shapes
                    count = -1
                for force in all_forces:                #plots arrows as forces
                    count += 1
                    if force.mag <= 1e-6:
                        continue
                    if np.floor(count / 8) % 3 == 0:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 500,
                                        force.mag * np.sin(force.ang) / 500, color='black', width=0.0005)
                    elif np.floor(count / 8) % 3 == 1:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 500,
                                        force.mag * np.sin(force.ang) / 500, color='orange', width=0.0005)
                    else:
                        bow = plt.arrow(force.xpos, force.ypos, force.mag * np.cos(force.ang) / 500,
                                        force.mag * np.sin(force.ang) / 500, color='green', width=0.0005)
                    plt.gca().add_patch(bow)
                plt.axis('scaled')
                plt.plot(string_arr[:, 0], string_arr[:, 1], color='red')
                plt.show()
            exte=args.R-beads[args.n_beads].B[1]            #plots the vertical displacement of the top right corner of the rightmost bead
            extension[m,j]=exte*1000
            for i in range(args.n_beads):
                fin_disp[m, i, j] = disp_ini[i]     #collects data for graphs
                fin_tht[m, i, j] = tht_ini[i]
    force=np.linspace(0,F,args.n_iters)
    #for i in range(args.param_sweep):                    #displacement plot
        #plt.plot(force,fin_disp[:,args.n_beads-1,i])
    #plt.legend(['0.011','0.013','0.015','0.017','0.019'])
    #plt.xlabel('Force [N]')
    #plt.ylabel('displacement [m]')
    #plt.show()
    #for i in range(args.param_sweep):                    #angle plot
        #plt.plot(force,fin_tht[:,args.n_beads-1,i])
    #plt.legend(['0.011','0.013','0.015','0.017','0.019'])
    #plt.xlabel('Force [N]')
    #plt.ylabel('Angular displacement [radians]')
    #plt.show()
    read_data=pandas.read_csv(r"Specimen_RawData_3.csv").to_numpy()
    read_ext = read_data[2:1203, 1].astype('float64')
    read_for = read_data[2:1203, 2].astype('float64')
    read_ext = np.multiply(read_ext,-1)
    read_for = np.multiply(read_for,-1)
    for x in range(0,1201):
        read_ext[x]-=1


    #plt.plot(read_for,read_ext)
    for i in range(args.param_sweep):            #extension plot
        plt.plot(force,extension[:,i])
    plt.xlabel('Force [N]')
    plt.ylabel('Vertical displacement [mm]')
    plt.legend(['5mm','6mm','7mm','8mm'])
    plt.show()
    fig=plt.figure()
    ani=animation.FuncAnimation(fig,animate,interval=50,blit=True)
    plt.show()
    f = r"c://Users/Chris/Desktop/animation.mp4"
    plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Chris\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'

    #ani.save(f,writer='ffmpeg')
main()
