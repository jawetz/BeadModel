import numpy as np

def intersect(bead_no, st_pt, out,beads,args):
    #calculates how the string will contact the bead
    end_pt=beads[args.n_beads].pos
    slope=(end_pt[1]-st_pt[1])/(end_pt[0]-st_pt[0])
    if out:
        fst_pt_bot = beads[bead_no + 1].E
        fst_pt_top = beads[bead_no + 1].C
    else:
        fst_pt_bot = beads[bead_no].F
        fst_pt_top = beads[bead_no].D
    if fst_pt_bot[1]>st_pt[1]+slope*(fst_pt_bot[0]-st_pt[0]):
        return 'bot'
    elif fst_pt_top[1]<st_pt[1]+slope*(fst_pt_top[0]-st_pt[0]):
        return 'top'
    else:
        return fst_pt_top[1]-st_pt[1]+slope*(fst_pt_top[0]-st_pt[0])