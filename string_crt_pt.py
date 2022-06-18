import numpy as np
from bead import Bead
from bead_pos import bead_pos
from no_overlap import max_disp
def two_pt_crt_pt(args):
    if args.thtb==np.pi/2:
        return 0
    beads = np.zeros(args.n_beads + 1, dtype=Bead)
    d = (args.R) / np.sin(args.thtb)
    beads[0]=Bead([0, 0], 0, 0, 0.015, args.R, args.r, args.thtb,args.mis)
    tht_ini = [0.0] * args.n_beads
    disp_ini = [0.00] * args.n_beads
    dist=1
    while np.abs(dist)>=0.0000000001:
        for i in range(1, args.n_beads + 1):
            new_pos = bead_pos(beads[i - 1], tht_ini[i - 1], disp_ini[i - 1],args)  # adds beads to the system
            new_bead = Bead(new_pos, tht_ini[i - 1], disp_ini[i - 1], args.w, args.R, args.r, args.thtb,args.mis)
            beads[i] = new_bead
        l=beads[0].F
        r=beads[args.n_beads].pos
        m=beads[1].C
        frac_over=(m[0]-l[0])/(r[0]-l[0])
        str_pt=(1-frac_over)*l[1]+frac_over*r[1]
        dist=m[1]-str_pt
        for j in range(args.n_beads):
            tht_ini[j]+=10*dist
        disp_ini[0]=max_disp(tht_ini[0],args)
    return tht_ini[0]


