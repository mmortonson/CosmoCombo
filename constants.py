"""
Definitions of constants useful for cosmology modules.

2013-05-06 Module created.
"""

import numpy as np
import scipy.constants

# conversion factors between physical and dimensionless units
# for Hubble expansion rate, distances, and related quantities
C_HUB_MPC = scipy.constants.c*1.0e-5 # c*h/H_0 in Mpc
C_HUB_GPC = scipy.constants.c*1.0e-8 # c*h/H_0 in Gpc
HUB_S = 10.*scipy.constants.parsec # h/H_0 in s

# factor of c^2 to convert g to erg (c in cm/s)
C_G_ERG = (scipy.constants.c*100.)**2

# radiation constant in erg cm^{-3} K^{-4}
RAD_CONST_ERG_CM3_K4 = 7.56577e-15

# critical density at present in h^2 g cm^{-3}
RHO_CRIT_H2_G_CM3 = 3./(8000.*np.pi*scipy.constants.G*HUB_S**2)

# present CMB temperature
T_CMB_K = 2.725

# photon density fraction at present, Omega_gamma*h^2
OMEGA_GAMMA_H2 = RAD_CONST_ERG_CM3_K4*T_CMB_K**4/(C_G_ERG*RHO_CRIT_H2_G_CM3)

# redshift of recombination
# !!!!! replace this with functions to calculate zrec in other modules
ZREC_APPROX = 1090.0

# present neutrino temperature
T_NU_K = (4./11.)**(1./3.) * T_CMB_K

# area of full sky in square degrees
FULL_SKY_SQ_DEG = 4.*np.pi*(180./np.pi)**2
