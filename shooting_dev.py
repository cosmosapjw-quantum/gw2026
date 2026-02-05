import numpy as np

from newton_raphson_dev import nrmethod2D
from rk2_heun_dev import h_c, phiddot, hdot, phi_ini_1, rkheunPoisson

def phidot_1(Rs):
    _, phi, phidot, _ = rkheunPoisson(phiddot,hdot,phi_ini_1,h_c,1e-5,0.5,500,Rs)
    return phi, phidot

def phidot_2(Rs,Ms):
    phi_ini_2 = [-Ms/Rs, Ms/Rs]
    _, phi, phidot, _ = rkheunPoisson(phiddot,hdot,phi_ini_2,0.0,1.0,0.5,500,Rs)
    return phi, phidot

def func(x):
    Rs, Ms = x

    phi_inter, phidot_inter = phidot_1(Rs)
    phi_outer, phidot_outer = phidot_2(Rs,Ms)

    phi_diff = (phi_inter - phi_outer) - (h_c + Ms/Rs)
    phidot_diff = phidot_inter - phidot_outer

    return np.stack([phi_diff, phidot_diff], axis=0)

def shooting(Rs_start,Ms_start):
    Rs,Ms = nrmethod2D(func, [Rs_start,Ms_start],1e-5)
    return Rs,Ms

print(shooting(1.0,1.0))