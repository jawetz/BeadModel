import numpy as np

def intersect(bead_no, st_pt, out,beads,args):
    #calculates how the string will contact the bead
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

    if st_pt[1] > fst_pt_top[1] + (fst_pt_top[1] - scd_pt_bot[1]) / (fst_pt_top[0] - scd_pt_bot[0]) * (
            st_pt[0] - fst_pt_top[0]):
        return 'top'
    elif st_pt[1] < fst_pt_bot[1] + (fst_pt_bot[1] - scd_pt_top[1]) / (fst_pt_bot[0] - scd_pt_top[0]) * (
            st_pt[0] - fst_pt_bot[0]):
        return 'bot'
    else:
        return 'none'

