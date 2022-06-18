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
        disp=(2*args.R*(np.cos(ang)+np.sin(ang)*np.tan(args.thtb)-1))/(np.sin(args.thtb-args.mis-ang)+np.cos(args.thtb-args.mis-ang)*np.tan(args.thtb))
    return disp



#ensures the beads dont overlap
def min_ang(disp,args):
    d = (args.R) / np.sin(args.thtb-args.mis)  # length of angled side
    if disp ==0:
        return 0
    if d<disp:
        print(disp)
    else:
        tht = np.atan((d * np.sin(args.thtb - args.mis)) / (d - disp-d*np.cos(args.thtb-args.mis)))
        return tht-args.thtb
