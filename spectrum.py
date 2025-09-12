import sys, os
import numpy as np

from scipy import interpolate as interp

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.constants as const

import healpy as hp
import vegas

# NPTFit_dir = '/global/cfs/cdirs/m3166/bsafdi/NPTFit/'
NPTFit_dir = '/clusterfs/heptheory/brsafdi/brsafdi/github/NPTFit/'
sys.path.append(NPTFit_dir)
from NPTFit import create_mask as cm

from NucleonCouplings import Ncouplings

# ================================================================= #



def integrator(fnc, interval):
    @vegas.lbatchintegrand
    def integrand(x):
        if x.shape[1] > 1:
            xx = []
            for i in range(x.shape[1]):
                xx.append(x[:,i])
        else:
            xx = x[:,0]
        return fnc(xx)
    integ = vegas.Integrator(interval)
    integ(integrand, nitn = 10, neval = 1e4)
    result = integ(integrand, nitn = 10, neval = 1e4)
    return result.mean



def load_time(LD_dir):
    filelist = os.listdir(LD_dir)
    time = np.zeros(len(filelist))

    for i in range(len(filelist)):
        filename = filelist[i]
        time[i] = filename.split('=')[1].split('.txt')[0]

    sort = np.argsort(time)
    return time[sort]



def load_spec(LD_dir, Eoutput = False):
    filelist = os.listdir(LD_dir)
    Ebin = np.concatenate((np.linspace(0.001,0.1,20),np.linspace(0.12,1.,30)))
    
    time = np.zeros(len(filelist))
    spectra = np.zeros((len(filelist), len(Ebin)))
    
    for i in range(len(filelist)):
        filename = filelist[i]
        time[i] = filename.split('=')[1].split('.txt')[0]
        spectra[i] = np.loadtxt(LD_dir + filename)
        
    sort = np.argsort(time)
    time_sorted = np.copy(time[sort])
    spectra_sorted = np.copy(spectra[sort])
    
    if Eoutput:
        return Ebin, spectra_sorted
    else:
        return spectra_sorted



def load_prim(LD_dir, Eoutput = False):
    filelist = os.listdir(LD_dir)
    if ".ipynb_checkpoints" in filelist:
        filelist.remove(".ipynb_checkpoints")

    Ebin = np.concatenate((np.linspace(0.001,0.1,20),np.linspace(0.12,1.,30)))

    time = np.zeros(len(filelist))
    spectra = np.zeros((len(filelist), len(Ebin)))

    for i in range(len(filelist)):
        filename = filelist[i]
        time[i] = filename.split('=')[1].split('.txt')[0]
        spectra[i] = np.loadtxt(LD_dir + filename)

    sort = np.argsort(time)
    time_sorted = np.copy(time[sort])
    spectra_sorted = np.copy(spectra[sort])

    if Eoutput:
        return Ebin, spectra_sorted
    else:
        return spectra_sorted



def load_brem(LD_dir, Eoutput = False):
    folders = ['Spectra_Can/', 'Spectra_Cap/', 'Spectra_CanCap/']

    Ebin = np.concatenate((np.linspace(0.001,0.1,20),np.linspace(0.12,1.,30)))

    filelist = np.copy(os.listdir(LD_dir + folders[0]))

    time_sorted = np.zeros((len(folders), len(filelist)))
    spectra_sorted = np.zeros((len(folders), len(filelist), len(Ebin)))

    for i in range(len(folders)):
        filelist = os.listdir(LD_dir + folders[i])
        if ".ipynb_checkpoints" in filelist:
            filelist.remove(".ipynb_checkpoints")

        time = np.zeros(len(filelist))
        spectra = np.zeros((len(filelist), len(Ebin)))

        for j in range(len(filelist)):
            filename = filelist[j]
            time[j] = filelist[j].split('=')[1].split('.txt')[0]
            spectra[j] = np.loadtxt(LD_dir + folders[i] + filename)

        sort = np.argsort(time)
        time_sorted[i] = np.copy(time[sort])
        spectra_sorted[i] = np.copy(spectra[sort])

    if Eoutput:
        return Ebin, spectra_sorted
    else:
        return spectra_sorted



def load_pion(LD_dir, pion_condensate = True, Eoutput = False):
    folders = ['Spectra_Can/', 'Spectra_Cap/', 'Spectra_CanCap/']

    Ebin = np.concatenate((np.linspace(0.001,0.1,20),np.linspace(0.12,1.,30)))
    zero_range = np.where(Ebin < 0.2)[0]

    filelist = os.listdir(LD_dir + folders[0])
    if ".ipynb_checkpoints" in filelist:
        filelist.remove(".ipynb_checkpoints")

    time_sorted = np.zeros((len(folders), len(filelist)))
    spectra_sorted = np.zeros((len(folders), len(filelist), len(Ebin)))

    for i in range(len(folders)):
        filelist = os.listdir(LD_dir + folders[i])
        if ".ipynb_checkpoints" in filelist:
            filelist.remove(".ipynb_checkpoints")
            
        time = np.zeros(len(filelist))
        spectra = np.zeros((len(filelist), len(Ebin)))

        for j in range(len(filelist)):
            filename = filelist[j]
            time[j] = filelist[j].split('=')[1].split('.txt')[0]
            loaded_spec = np.loadtxt(LD_dir + folders[i] + filename)
            if pion_condensate is False:
                print('pion condensate removed')
                loaded_spec[zero_range] = 0
            spectra[j] = loaded_spec

        sort = np.argsort(time)
        time_sorted[i] = np.copy(time[sort])
        spectra_sorted[i] = np.copy(spectra[sort])

    if Eoutput:
        return Ebin, spectra_sorted
    else:
        return spectra_sorted
    

def calc_smm_spec(LD_dir, ma_array, smm_energy, convprob):
    Ebins, loaded = load_spec(LD_dir, Eoutput = True)
    spec_list = []
    
    for int_I in smm_energy:
        tm_spec = np.zeros((len(ma_array), loaded.shape[0]))
        
        for ti in range(loaded.shape[0]):
            for mi in range(len(ma_array)):
                photon_spec = interp.interp1d(Ebins, loaded[ti] * convprob[mi])
                tm_spec[mi, ti] = integrator(photon_spec, [int_I])
        spec_list.append(tm_spec)
        
    return spec_list