import numpy as np


#ensures beads dont overlap
def max_disp(ang,args):
    d = (args.R) / np.sin(args.thtb)  # length of angled side
    ang3 = np.pi - ang - 2 * args.thtb
    if args.thtb == np.pi / 2:
        return args.R
    elif ang == 0:
        return 0
    else:
        disp = d - np.sin(ang3) * d / np.sin(2 * args.thtb)
    return disp



#ensures the beads dont overlap
def min_ang(disp,args):
    d = (args.R) / np.sin(args.thtb)  # length of angled side
    if disp ==0:
        return 0
    if d<disp:
        print(disp)
    if np.abs((d-disp)*np.sin(2*args.thtb)/d) >=1:
        return np.pi/2 - 2*args.thtb
    else:
        ang3 = np.arcsin((d - disp) * np.sin(2 * args.thtb) / d)
    if (np.sin(2*args.thtb) < (d-disp)/d) and args.thtb<=np.pi/4:
        return np.abs(2 * args.thtb - ang3)
    else:
        return np.pi-2 * args.thtb - ang3
