import numpy as np

def surface_contact(vars, contact_forces):
    x, y = vars
    sum_x = 0
    sum_y = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
    eq1 = sum_x + x * np.cos(contact_forces[-2].ang) + y * np.cos(contact_forces[-1].ang)
    eq2 = sum_y + x * np.sin(contact_forces[-2].ang) + y * np.sin(contact_forces[-1].ang)
    return eq1, eq2
def point_contact(vars, contact_forces):
    x, y = vars
    sum_x = 0
    sum_y = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
    eq1 = sum_x + x * np.cos(contact_forces[-2].ang) + y * np.cos(contact_forces[-1].ang)
    eq2 = sum_y + x * np.sin(contact_forces[-2].ang) + y * np.sin(contact_forces[-1].ang)
    return eq1, eq2


def two_point_contact(vars, contact_forces):
    x, y = vars
    sum_x = 0
    sum_y = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
    eq1 = sum_x + x * np.cos(contact_forces[-2].ang) + y * np.cos(contact_forces[-1].ang)
    eq2 = sum_y + x * np.sin(contact_forces[-2].ang) + y * np.sin(contact_forces[-1].ang)
    return eq1, eq2

def positioning(vars, mome, surface_top,surface_bot,bead):
    x = vars
    eq1 = mome - np.cos(surface_top.ang) * (surface_top.ypos - x*np.sin(bead.thtb+bead.theta)-bead.E[1]) * surface_top.mag + np.sin(
        surface_top.ang) * surface_top.mag * (surface_top.xpos + x*np.cos(bead.thtb+bead.theta)-bead.E[0])-np.cos(surface_bot.ang) * (surface_bot.ypos - x*np.sin(bead.thtb-bead.theta)-bead.E[1]) * surface_bot.mag + np.sin(
        surface_bot.ang) * surface_bot.mag * (surface_bot.xpos - x*np.cos(bead.thtb-bead.theta)-bead.E[0])
    return eq1

def friction(vars,contact_forces,bead,args):
    s,t,u = vars
    sum_x = 0
    sum_y = 0
    sum_M = 0
    for force in contact_forces:
        sum_x += force.mag * np.cos(force.ang)
        sum_y += force.mag * np.sin(force.ang)
        sum_M += -force.mag * np.cos(force.ang) * (force.ypos - bead.E[1]) + force.mag * (
                    force.xpos - bead.E[0]) * np.sin(force.ang)
    if args.thtb>=np.pi/4:
        eq1 = sum_x + s * np.cos(contact_forces[-2].ang+u) + t * np.cos(contact_forces[-1].ang+u)
        eq2 = sum_y + s * np.sin(contact_forces[-2].ang+u) + t * np.sin(contact_forces[-1].ang+u)
        eq3 = sum_M - s * np.cos(contact_forces[-2].ang+u) * (contact_forces[-2].ypos - bead.E[1]) + s * (
                    contact_forces[-2].xpos - bead.E[0]) * np.sin(contact_forces[-2].ang+u) - t * np.cos(
            contact_forces[-1].ang+u)*(contact_forces[-1].ypos-bead.E[1]) + t*(contact_forces[-1].xpos-bead.E[0]) * np.sin(contact_forces[-1].ang+u)
    else:
        eq1 = sum_x + s * np.cos(contact_forces[-2].ang + u) + t * np.cos(contact_forces[-1].ang - u)
        eq2 = sum_y + s * np.sin(contact_forces[-2].ang + u) + t * np.sin(contact_forces[-1].ang - u)
        eq3 = sum_M - s * np.cos(contact_forces[-2].ang + u) * (contact_forces[-2].ypos - bead.E[1]) + s * (
                contact_forces[-2].xpos - bead.E[0]) * np.sin(contact_forces[-2].ang+u) - t * np.cos(
            contact_forces[-1].ang - u) * (contact_forces[-1].ypos - bead.E[1]) + t * (
                          contact_forces[-1].xpos - bead.E[0]) * np.sin(contact_forces[-1].ang - u)
    return eq1, eq2, eq3