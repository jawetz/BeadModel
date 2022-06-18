from main import shapes,beads
from bead import Bead
def init():
    shapes.set_data([],[])
    return shapes
def animate(i):

    shapes = []  # for making the pictures
    # for this bead and all beads to the right
    # add displacement, angle around the contact point
    beads[0] = Bead([0, 0], 0, 0, w_b, R, r, thtb)
    for j in range(n_beads):
        new_pos = bead_pos(beads[j], fin_tht[i,j,0], fin_disp[i,j,0])  # adds beads to the system
        new_bead = Bead(new_pos, fin_tht[i,j,0], fin_disp[i,j,0], w, R, r, thtb)
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

