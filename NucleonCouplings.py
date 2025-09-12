from IPython.display import clear_output
import time, sys, os

import numpy as np
from astropy.io import fits
from astropy import units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
#import healpy as hp
import scipy.special as sp
import scipy.integrate as itg
# import vegas
import multiprocessing

import csv
import numpy as np
from scipy import interpolate

from numpy import loadtxt
import time


######################
# defining constants #
######################
pi = 3.14
mTOMeVm1 = 5.06*1.E12
MeVTOsm1 = 1.52*1.E21
grToMeV = 5.62*1.E26
cmm1ToMeV = 1.98*1.E-11

mpi = 135. 
mNucleon = 938.
fpi = 92.4
gAx = 1.26
Gamma_Delta = 117.
mDelta = 1232.
alpha =1./137.

def Ncouplings(x1, x2):   #x1 in GeV^-1  #x2 in GeV^-1
    
    gagamma = x1
    CBofa = x2 # CB/fa
    CWofa = gagamma * (2.*pi)/ alpha - CBofa
    Lambda = 2.*alpha/gagamma
    Cgamma_o_fa = CBofa + CWofa
    mNucleon = 0.938
    mZ = 90.
    
    sinTheta_mZ = (0.231)**0.5;
    cosTheta_mZ = (1 - 0.231)**0.5;
    ee_mZ = 0.309119
    ee_2 = 0.306146
    g1_mZ = (4.*pi/127)**0.5/cosTheta_mZ
    g2_mZ = (4.*pi/127)**0.5/sinTheta_mZ

    def g1(mu):
        return (4 * 3**0.5 * pi * g1_mZ)/((48. * pi**2 + 41.*np.log(mZ/mu)*g1_mZ**2)**0.5)
    
    def g2(mu):
        return (4 * 3**0.5 * pi * g2_mZ)/((48. * pi**2 + 19.*np.log(mu/mZ)*g2_mZ**2)**0.5)
    
    def Cquark(YQ,Yq,Q):
        CqmZ = 3./(128. * pi**4) * np.log(Lambda/mZ) * (3./4. * CWofa * g2(Lambda)**2 * g2_mZ**2 + (YQ**2+Yq**2) * CBofa * g1(Lambda)**2 * g1_mZ**2)
        Cqalpha = 3./(64. * pi**4) * np.log(mZ/2.) * Q**2 * ee_mZ**2 * ee_2**2 * Cgamma_o_fa
        return CqmZ + Cqalpha
    
    Cu_o_fa = Cquark(1./6.,2./3.,2./3.)
    Cd_o_fa = Cquark(1./6.,-1./3.,-1./3.)
    Cs_o_fa = Cquark(1./6.,-1./3.,-1./3.)
    
    Cp_o_fa = 0.897 * Cu_o_fa - 0.376 * Cd_o_fa - 0.026 * Cs_o_fa 
    
    Cn_o_fa = 0.897 * Cd_o_fa - 0.376 * Cu_o_fa - 0.026 * Cs_o_fa
    
    NCoup = [mNucleon * Cp_o_fa, mNucleon * Cn_o_fa]
    
    return NCoup