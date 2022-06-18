import numpy as np

def bead_pos(prev_bead, tht, disp,args):
    #calculates the position of the tip of the next bead
    if disp >= 0:
        contact_pt = prev_bead.H
        back_corner = contact_pt - np.array([disp * np.cos(prev_bead.thtb - tht), disp * np.sin(prev_bead.thtb - tht)])
    else:
        back_corner = prev_bead.H + np.array([-disp * np.cos(prev_bead.thtb), -disp * np.sin(prev_bead.thtb)])
    front_corner = back_corner + np.array([args.w * np.cos(tht), -args.w * np.sin(tht)])
    pos = front_corner + np.array([(prev_bead.R / np.sin(prev_bead.thtb)) * np.cos(prev_bead.thtb - tht), (prev_bead.R / np.sin(prev_bead.thtb)) * np.sin(prev_bead.thtb - tht)])
    return pos