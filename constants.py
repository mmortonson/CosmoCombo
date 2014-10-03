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

# redshift of recombination
# !!!!! replace this with functions to calculate zrec in other modules
ZREC_APPROX = 1090.0

# present CMB temperature
T_CMB_K = 2.725

# present neutrino temperature
T_NU_K = (4./11.)**(1./3.) * T_CMB_K

# area of full sky in square degrees
FULL_SKY_SQ_DEG = 4.*np.pi*(180./np.pi)**2
