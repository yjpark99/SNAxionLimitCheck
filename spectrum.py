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



def load_effA(LD_dir):
    with fits.open(LD_dir + 'exp.fits') as f:
        tstart = f[0].header['TSTART']
        tstop = f[0].header['TSTOP']
        
        Ebins = f[2].data['energy'] / 1e3 # units of GeV
        expmap = np.zeros(len(Ebins))
        
        for ei in range(len(expmap)):
            expmap[ei] = np.amax(f[1].data['energy' + str(ei + 1)] / (tstop - tstart)) # units of cm^2
            
    return Ebins, expmap



def load_bkg(ell, b, distance, effA_dir, mission_dir):
    coord = SkyCoord(l = ell * u.deg, b = b * u.deg, frame = 'galactic')
    
    with fits.open(mission_dir + 'counts.fits') as f:
        emin = f[2].data['E_MIN'] / 1e6
        emax = f[2].data['E_MAX'] / 1e6
        nside = f[1].header['nside']
        
        skymap = np.zeros((len(emin), hp.nside2npix(nside)))
        for ei in range(len(emin)):
            skymap[ei] = f[1].data['CHANNEL' + str(ei + 1)]
            
    counts = np.zeros(len(emin))
    mask = cm.make_ring_mask(0, 5, coord.b.value, coord.l.value, nside)
    
    for ci in range(len(counts)):
        mask_map = hp.ma(skymap[ci])
        mask_map.mask = mask
        counts[ci] = np.sum(mask_map.compressed())
        
    mission_amax = []
        
    with fits.open(mission_dir + 'exp.fits') as f:
        exp_E = f[2].data['ENERGY']
        dt = f[0].header['TSTOP'] - f[0].header['TSTART']
        for ei in range(len(exp_E)):
            expmap = hp.ma(f[1].data['ENERGY' + str(ei + 1)])
            expmap.mask = mask
            mission_amax.append(np.argmax(expmap))
            
        mission_amax = np.unique(mission_amax)
        missionA = np.zeros(len(exp_E) - 1)
        
        for eii in range(len(exp_E) - 1):
            expmap = np.mean((f[1].data['ENERGY' + str(eii + 1)], f[1].data['ENERGY' + str(eii + 2)]), axis = 0)
            missionA[eii] = expmap[mission_amax[0]] / dt
            
        bkg_rate = counts / dt
        
        Ebins, max_effA = load_effA(effA_dir)
        f_maxA = interp.interp1d(Ebins, max_effA, bounds_error = False, fill_value = 'extrapolate')
        bkg_rate_rescale = bkg_rate * f_maxA((emin + emax) / 2) / np.mean(missionA, axis=0)
        
        return bkg_rate_rescale

    

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