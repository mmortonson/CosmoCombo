"""
Classes for cosmological models. The basic class defines parameters and 
methods for Lambda CDM. Subclasses add parameters to extend the model
and/or modify the methods (e.g., changing the dark energy model).

2013-05-06 Module created.
           Copied from ~/Projects/PlanckDE/Code/planckde.ipynb .
           Changed base class name from 'cosmology' to 'lcdm'.
2013-05-07 Added pk_nl_cosmicemu functions.
2013-06-19 Added volume function.
2013-07-19 Added comoving distance function, modified angular diameter distance.
2014-01-15 Added neutrino mass parameter and new sound horizon 
           fitting formula (sound_horizon_drag_anderson13).
2014-01-22 Changed radiation density to allow for massive neutrinos.
           Added photon density parameter omegagamma and effective number 
           of neutrino species neff.
           Added neutrino density method and modified hubble method.
2014-01-29 Added optional sq_deg argument to volume method.
2014-03-14 Fixed bug in omegar - massless neutrino contribution was 
           missing a factor of 1/T_CMB_K**4.
2014-06-02 Replaced sqrt, exp, log imported from math with numpy version.
2014-10-03 Changed base class name from 'lcdm' to 'BaseModel'.
"""

import numpy as np
from scipy import integrate
from scipy import interpolate

import subprocess
import StringIO
from collections import OrderedDict

from constants import *
import utils


model_class = None
parameters = []

# add any cosmology functions here that you want to use as derived parameters

def DA_BOSS_DR11(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    m_fid = LCDM(omegam=0.274, h=0.70, omegabhh=0.0224, 
                 ns=0.95, sigma8=0.8, mnu=0.0)
    sound_horizon_ratio = m.sound_horizon_drag_anderson13()/ \
        m_fid.sound_horizon_drag_anderson13()
    return C_HUB_MPC*m.dist_ang_diam(z)/(1.+z)/sound_horizon_ratio

def H_BOSS_DR11(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    m_fid = LCDM(omegam=0.274, h=0.70, omegabhh=0.0224, 
                 ns=0.95, sigma8=0.8, mnu=0.0)
    sound_horizon_ratio = m.sound_horizon_drag_anderson13()/ \
        m_fid.sound_horizon_drag_anderson13()
    return 100.*m.hubble(z)*sound_horizon_ratio

def DV_BOSS_DR11(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    m_fid = LCDM(omegam=0.274, h=0.70, omegabhh=0.0224, 
                 ns=0.95, sigma8=0.8, mnu=0.0)
    sound_horizon_ratio = m.sound_horizon_drag_anderson13()/ \
        m_fid.sound_horizon_drag_anderson13()
    return C_HUB_MPC*m.dv_bao(z)/sound_horizon_ratio

def rd_fid_BOSS_DR11(*values):
    m_fid = LCDM(omegam=0.274, h=0.70, omegabhh=0.0224, 
                 ns=0.95, sigma8=0.8, mnu=0.0)
    return m_fid.sound_horizon_drag_anderson13()

def DV_SDSS_DR7_MGS(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    m_fid = LCDM(omegam=0.31, h=0.67, omegabhh=0.02155, 
                 ns=0.96, sigma8=0.83, mnu=0.0)
    sound_horizon_ratio = m.sound_horizon_drag_anderson13()/ \
        m_fid.sound_horizon_drag_anderson13()
    return C_HUB_MPC*m.dv_bao(z)/sound_horizon_ratio

def mu_SN(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    dl = C_HUB_MPC*(1.+z)*m.dist_ang_diam(z)
    return 5.*np.log10(dl) + 25.

def f_sigma8(z, *values):
    m = model_class(**dict(zip(parameters, values)))
    return m.growth_rate_gamma(z)*m.sigma8_z_gamma(z)


functions = {'DA_BOSS_DR11': np.vectorize(DA_BOSS_DR11),
             'H_BOSS_DR11': np.vectorize(H_BOSS_DR11),
             'DV_BOSS_DR11': np.vectorize(DV_BOSS_DR11),
             'rd_fid_BOSS_DR11': np.vectorize(rd_fid_BOSS_DR11),
             'DV_SDSS_DR7_MGS': np.vectorize(DV_SDSS_DR7_MGS),
             'mu_SN': np.vectorize(mu_SN),
             'f_sigma8': np.vectorize(f_sigma8)}


class BaseModel(object):
    """ Parameters and functions for basic LCDM cosmology. """
    def __init__(self, **kwargs):
# !!!!! modify this so it's easy to change the default parameters
# !!!!! to some pre-defined model (e.g. Planck or WMAP-X best fit)
        self.h = kwargs.get('h', 0.6711)
        self.omegam = kwargs.get('omegam', 0.3175)
        self.omegamhh = self.omegam*self.h**2
        self.omegabhh = kwargs.get('omegabhh', 0.022068)
        self.omegagamma = kwargs.get('omegagamma', 2.469e-5/(self.h**2))
        #self.omegar = 4.17e-5/(self.h**2) # radiation density including 3.046 neutrinos
        self.mnu = kwargs.get('mnu', 0.0) # sum of neutrino masses in eV
        self.omeganuhh = self.mnu/93.14
        self.neff = kwargs.get('neff', 3.046) # effective number of neutrinos
        self.omegar = self.omegagamma
        if self.mnu == 0.0:
            self.omegar *= 1. + (7./8.)*(T_NU_K/T_CMB_K)**4*self.neff
        self.omegak = kwargs.get('omegak', 0.0)
        self.omegade = 1.0 - self.omegam - self.omegar - self.omegak
        self.sigma8 = kwargs.get('sigma8', 0.8344)
        self.ns = kwargs.get('ns', 0.96)
        self.w = -1.0
        # define spline variables/dictionaries
        self.pk_nl_cosmicemu_spl = {}
        # add non-standard parameters
        self.init_extra_params(**kwargs)
        
    def init_extra_params(self, **kwargs):
        """ Method used by subclasses to add new parameters at the end of __init__. """
        pass

# !!!!! add methods to modify parameter values

    def de_density(self, z):
        """ Dark energy density at z divided by the present critical density. """
        return self.omegade

    def d_de_density_dlna(self, z):
        """ Derivative of de_density w.r.t. ln(a). """
        return 0.0

    def neutrino_density(self, z):
        """ Neutrino density at z divided by the present critical density. 
            Uses fitting formula for transition between relativistic and
            nonrelativistic neutrinos from Komatsu+ 1001.4538 sec. 3.3. """
        A = 0.3173
        p = 1.83
        y = 187.e3*self.omeganuhh/(1.+z)
        f_nu_rel_to_nonrel = (1. + (A*y)**p)**(1./p)
        return self.omegagamma*(1.0+z)**4*(7./8.)*T_NU_K**4*self.neff* \
                   f_nu_rel_to_nonrel
        
    def hubble(self, z):
        """ Dimensionless Hubble parameter H(z) (divide by C_HUB_* to get units of 1/*). """
        return self.h*np.sqrt(self.omegam*(1.0+z)**3 + self.omegak*(1.0+z)**2 +\
                                  self.omegagamma*(1.0+z)**4 + \
                                  self.neutrino_density(z) + self.de_density(z))
    
    def dist_com(self, z):
        """ Dimensionless, comoving distance (multiply by C_HUB_* to get units of *). """
        d, err = integrate.quad(lambda x: 1.0/self.hubble(x), 0.0, z)
        return d

    def dist_ang_diam(self, z):
        """ Dimensionless, comoving, angular diameter distance D_A(z) (multiply by C_HUB_* to get units of *). """
        y = self.dist_com(z)
        K = -self.omegak*self.h**2
        if K == 0:
            d = y
        elif K > 0:
            d = np.sin(np.sqrt(K)*y)/np.sqrt(K)
        elif K < 0:
            d = np.sinh(np.sqrt(-K)*y)/np.sqrt(-K)
        return d
    
    def volume(self, z1, z2, sq_deg=FULL_SKY_SQ_DEG):
        """ Comoving volume between z1 and z2 
        (multiply by cube of C_HUB_* to get units of (*)^3). 
        Optional argument sq_deg specifies area in square degrees with sq_deg
        (default is full sky).
        """
        y, err = integrate.quad(lambda x: \
                                self.dist_ang_diam(x)**2/self.hubble(x), z1, z2)
        return 4.0*np.pi*(sq_deg/FULL_SKY_SQ_DEG)*y

    def dv_bao(self, z):
        """ Dimensionless D_V(z), BAO combination of D_A(z) and H(z) (multiply by C_HUB_* to get units of *). """
        return (z*self.dist_ang_diam(z)**2/self.hubble(z))**(1.0/3.0)
    
    def growth_func_ode(self, lna, g_derivs):
        z = np.exp(-lna)-1.0
        omz = self.omegam*(1.0+z)**3*(self.h/self.hubble(z))**2
        dlnHdlna = 0.5*(self.h/self.hubble(z))**2 * \
            (-3.0*self.omegam*np.exp(-3.0*lna)-4.0*self.omegar*np.exp(-4.0*lna)\
             -2.0*self.omegak*np.exp(-2.0*lna)+self.d_de_density_dlna(lna))
        return [g_derivs[1], \
                    -(4.0 + dlnHdlna)*g_derivs[1] \
                    - (3.0 + dlnHdlna - 1.5*omz)*g_derivs[0]]

    def growth_rate(self, z):
        """ integrate ODE for growth function to find growth rate f. """
        if self.mnu != 0.0:
            print 'WARNING: effect of massive neutrinos on growth rate not included.'
        g_derivs0, lna0 = [1.0, 0.0], np.log(self.omegar/self.omegam)
        g_ode = integrate.ode(self.growth_func_ode).set_integrator('dopri5')
        g_ode.set_initial_value(g_derivs0, lna0)
        g_ode.integrate(-np.log(1.0+z))
        return g_ode.y[1]/g_ode.y[0] + 1.0

    def growth_rate_gamma(self, z):
        """ Approximate growth rate given by f = (Omega_m(z))**gamma. """
        gamma = 0.55
        return (self.omegam*(1.0+z)**3*(self.h/self.hubble(z))**2)**gamma

    def growth_rate_carroll(self, z):
        """ Flat LCDM growth rate using Carroll et al. growth function approximation. """
        omz = self.omegam*(1.0+z)**3*(self.h/self.hubble(z))**2
        gz = 2.5*omz/(omz**(4.0/7.0)-omz**2/140.0+(209.0/140.0)*omz+1.0/70.0)
        dgdom = gz/omz - 2.0*gz*gz/(5.0*omz)*(4.0/(7.0*omz**(3.0/7.0)) - \
                                              omz/70.0 + 209.0/140.0)
        domda = -3.0*omz**2*(1.0-self.omegam)/(self.omegam*(1.0+z)**2)
        return 1.0 + dgdom*domda/gz/(1.0+z)

    def sigma8_z(self, z):
        """ integrate ODE for growth function to find sigma_8(z). """
        if self.mnu != 0.0:
            print 'WARNING: effect of massive neutrinos on growth function not included.'
        g_derivs0, lna0 = [1.0, 0.0], np.log(self.omegar/self.omegam)
        g_ode = integrate.ode(self.growth_func_ode).set_integrator('dopri5')
        g_ode.set_initial_value(g_derivs0, lna0)
        g_ode.integrate(-np.log(1.0+z))
        gz = g_ode.y[0]
        g_ode.integrate(0.0)
        g0 = g_ode.y[0]
        return self.sigma8*gz/g0/(1.0+z)

    def sigma8_z_gamma(self, z):
        """ sigma_8 as a function of z using approximate growth_rate_gamma. """
        y, err = integrate.quad(lambda x: self.growth_rate_gamma(x)/(1.0+x), 0.0, z)
        return self.sigma8*np.exp(-y)

    def sigma8_z_carroll(self, z):
        """ sigma_8 as a function of z using Carroll et al. growth function approximation and assuming flat LCDM. """
        omz = self.omegam*(1.0+z)**3*(self.h/self.hubble(z))**2
        gz = 2.5*omz/(omz**(4.0/7.0)-omz**2/140.0+(209.0/140.0)*omz+1.0/70.0)
        om0 = self.omegam
        g0 = 2.5*om0/(om0**(4.0/7.0)-om0**2/140.0+(209.0/140.0)*om0+1.0/70.0)
        return self.sigma8*gz/g0/(1.0+z)

    def sound_horizon_rec_hu05(self):
        """ Approximate formula for sound horizon at recombination (in phys. Mpc) from Hu (2005). """
        return 144.4*(self.omegam*self.h**2/0.14)**(-0.252)*(self.omegabhh/0.024)**(-0.083)
    
    def sound_horizon_drag_percival09(self):
        """ Approximate formula for sound horizon at drag epoch (in phys. Mpc) from Percival et al. (2009). """
        return 153.5*(self.omegam*self.h**2/0.1326)**(-0.255)*(self.omegabhh/0.02273)**(-0.134)

    def sound_horizon_drag_anderson13(self):
        """ Approximate formula for sound horizon at drag epoch (in phys. Mpc) 
            from Anderson et al. (2013), 1312.4877. """
        return 55.234*(self.omegamhh-self.omeganuhh)**(-0.2538)* \
            self.omegabhh**(-0.1278)*(1.0+self.omeganuhh)**(-0.3794)
        
    def theta_rec_hu05(self):
        """ Angular scale of sound horizon at recombination, using Hu (2005) formula. """
        return self.sound_horizon_rec_hu05()/(C_HUB_MPC*self.dist_ang_diam(ZREC_APPROX))

    def pk_nl_cosmicemu(self, k, z):
        """ Interpolate P(k) using k and P values computed 
            by setup_pk_nl_cosmicemu. 
            Units: k [h/Mpc], P [(Mpc/h)**3]
        """
        # convert k to 1/Mpc units
        kh = k*self.h
        # check whether spline exists for this redshift;
        # if not, create new spline
        if z not in self.pk_nl_cosmicemu_spl:
            self.setup_pk_nl_cosmicemu(z)
        # make sure that k is in the allowed interpolation range
        spl_pts = self.pk_nl_cosmicemu_spl[z].get_knots()
        s = 'pk_nl_cosmicemu: k outside range ({0:g} h/Mpc, {1:g} h/Mpc)'
        assert spl_pts[0] <= kh <= spl_pts[-1], \
               s.format(spl_pts[0]/self.h, spl_pts[-1]/self.h)
        # return P(k) in (Mpc/h)**3 units
        return self.pk_nl_cosmicemu_spl[z](kh)*self.h**3

    def setup_pk_nl_cosmicemu(self, z):
        """ Use Cosmic Emulator v2.0 (a.k.a. FrankenEmu; 
            see Heitmann et al. 2013) to compute nonlinear P(k).
        """
        utils.check_ranges('cosmicemu',
                           {'z': [z, 0, 4],
                            'Omega_b*h**2': [self.omegabhh, 0.0215, 0.0235],
                            'Omega_m*h**2': [self.omegamhh, 0.120, 0.155],
                            'n_s': [self.ns, 0.85, 1.05],
                            'h': [self.h, 0.55, 0.85],
                            'w': [self.w, -1.3, -0.7],
                            'sigma_8': [self.sigma8, 0.61, 0.9]
                            })
        path = '/home/mmortonson/Code/CosmicEmu_v2.0'
        k = []
        p = []
        params = [str(self.omegabhh), str(self.omegamhh), \
                  str(self.ns), str(self.h*100.0), str(self.w), \
                  str(self.sigma8), str(z)]
        emu_output = subprocess.check_output([path + "/python_emu"] + params)
        reader = StringIO.StringIO(emu_output)
        for line in reader:
            pk = line.strip().split()
            k.append(float(pk[0]))
            p.append(float(pk[1]))
        self.pk_nl_cosmicemu_spl[z] = interpolate.UnivariateSpline(k, p, s=0)


class LCDM(BaseModel):
    """ Lambda CDM (alias for BaseModel class) """
    pass
    
class wCDM(BaseModel):
    """ Models with constant-w dark energy. """
    def init_extra_params(self, **kwargs):
        self.w = kwargs.get('w', -1.0)
    
    def de_density(self, z):
        return self.omegade*(1.0+z)**(3.0*(1.0+self.w))
    
class w0wa(BaseModel):
    """ Models with w_0-w_a dark energy. """
    def init_extra_params(self, **kwargs):
        self.w0 = kwargs.get('w0', -1.0)
        self.wa = kwargs.get('wa', 0.0)
    
    def de_density(self, z):
        return self.omegade*(1.0+z)**(3.0*(1.0+self.w0+self.wa))*np.exp(-3.0*self.wa*z/(1.0+z))

class w_zbins(BaseModel):
    """ Models where dark energy has piecewise constant w in bins of redshift:
            w(0 < z < zbin[0]) = w[0]
            w(zbin[0] < z < zbin[1]) = w[1]
            ...
            w(zbin[n-1] < z < infinity) = w[n]
    """
    def init_extra_params(self, **kwargs):
        self.zb = kwargs.get('zbins', [1.0])
        self.zb.insert(0, 0.0) # add redshift 0 to beginning of z bin list
        self.w = kwargs.get('w', [-1.0, -1.0])
        assert len(self.w) == len(self.zb), 'w_zbins: mismatch between length of w list and length of z bin list.'
        assert self.zb == sorted(self.zb), 'w_zbins: z bin list must be in increasing order.'
    
    def de_density(self, z):
        density = self.omegade
        for i in range(1, len(self.zb)):
            z1 = self.zb[i-1]
            z2 = self.zb[i]
            ilast = i
            if z > z2:
                density = density * ((1.0+z2)/(1.0+z1))**(3.0*(1.0+self.w[i-1]))
            else:
                ilast -= 1
                break
        density = density * ((1.0+z)/(1.0+self.zb[ilast]))**(3.0*(1.0+self.w[ilast]))
        return density
    
class EDE_DR06(BaseModel):
    """ Early dark energy models using the formula of Doran & Robbers (2006). """
    def init_extra_params(self, **kwargs):
        self.w0 = kwargs.get('w0', -1.0)
        self.omegaearly = kwargs.get('omegaearly', 0.01)
        if self.omegak != 0.0:
            print 'WARNING: Doran & Robbers (2006) early dark energy formula assumes a flat universe.'
    
    def de_density(self, z):
        de_frac = (self.omegade - self.omegaearly*(1.0-(1.0+z)**(3.0*self.w0))) / \
                  (self.omegade + self.omegam*(1.0+z)**(-3.0*self.w0)) + \
                  self.omegaearly*(1.0 - (1.0+z)**(3.0*self.w0))
        # should modify this for massive neutrinos - use hubble function?
        return (self.omegam*(1.0+z)**3 + self.omegar*(1.0+z)**4) / (1.0/de_frac - 1.0)

