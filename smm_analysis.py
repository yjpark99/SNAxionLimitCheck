import sys, os
import numpy as np

import scipy.interpolate as interp
from scipy import integrate
from scipy.special import xlogy

from astropy import units as u

sys.path.append('/global/scratch/users/yujinp/github/SNAxion/python/')
from NucleonCouplings import *

import vegas


# =================================================================================== #

def integrator(func, int_range):
    @vegas.batchintegrand
    def integrand(x):
        xx = x[:,0]
        return func(xx)
    integ = vegas.Integrator([int_range])
    integ(integrand, nitn = 10, neval = 1e5)
    result = integ(integrand, nitn = 10, neval = 1e5)
    return result.mean

# =================================================================================== #

class model:
    def __init__(self, ma_idx, dist, data_time, spec_dict, prim_flag = True, brem_flag = True, pion_flag = True, CapCan = None, CW_CB = 1):
        self.data_time = data_time
        self.spec_dict = spec_dict
        
        self.t_nu = 27341.37
        self.ma_idx = ma_idx
        self.CapCan = CapCan
        self.CW_CB = CW_CB
        
        self.ma_idx = ma_idx
        
        self.prim_flag = prim_flag
        self.brem_flag = brem_flag
        self.pion_flag = pion_flag
        
        self.Aeff = np.array([28, 115, 63])
        self.d = dist
        
        self._load_spec(self.ma_idx, self.spec_dict)
        
        if self.prim_flag:
            print('including primakoff')
            
        if self.brem_flag:
            print('including nucleon')
            
        if self.pion_flag:
            print('including pion')
        
        
    
    def _load_spec(self, ma_idx, _dict):
        self.time_bins = _dict['time_bins']
        self.prim = _dict['Primakoff'][:, ma_idx, :]
        self.brem_can = _dict['n_Can'][:, ma_idx, :]
        self.brem_cap = _dict['n_Cap'][:, ma_idx, :]
        self.brem_cancap = _dict['n_CanCap'][:, ma_idx, :]
        self.pion_can = _dict['p_Can'][:, ma_idx, :]
        self.pion_cap = _dict['p_Cap'][:, ma_idx, :]
        self.pion_cancap = _dict['p_CanCap'][:, ma_idx, :]
        
        
        
    def bkg_model(self, a):
        data_time = self.data_time
        t_nu = self.t_nu
        
        model = a[0] + a[1] * (data_time - t_nu) / 2.048
        model = np.vstack((model, a[2] + a[3] * (data_time - t_nu) / 2.048))
        model = np.vstack((model, a[4] + a[5] * (data_time - t_nu) / 2.048))
        return model
    
    
    
    def sig_model(self, gagg = 1e-12, c_var = 1):
        Aeff = self.Aeff
        d = self.d
        
        t_array = self.data_time
        t_nu = self.t_nu

        t_edges = np.zeros(len(t_array) + 1)
        t_edges[1:-1] = (t_array[1:] + t_array[:-1]) / 2
        t_edges[0] = t_array[0] - (t_array[1] - t_array[0]) / 2
        t_edges[-1] = t_array[-1] + (t_array[-1] - t_array[-2]) / 2

        bin_idx = np.searchsorted(t_edges, t_nu) - 1
        upper_bin_idx = np.searchsorted(t_edges, t_nu + 10)
        toi = np.where((np.arange(len(t_array)) <= upper_bin_idx) & (np.arange(len(t_array)) >= bin_idx))[0]
        
        model = np.zeros((3, len(t_array)))
        
        for ei in range(self.prim.shape[0]):
            
            for ti in range(len(toi) - 1):
                spec = 0
                start_T = 0 if (t_edges[toi[ti]] - t_nu) < 0 else t_edges[toi[ti]] - t_nu
                end_T = 10 if (t_edges[toi[ti + 1]] - t_nu) > 10 else t_edges[toi[ti + 1]] - t_nu
                print(start_T, end_T)
                
                if self.prim_flag:
                    prim_func = interp.interp1d(self.time_bins, self.prim[ei])
                    
                    prim_photon = gagg**2 * (gagg / 1e-12)**2 * integrator(prim_func, [start_T, end_T])
                    spec += prim_photon
                    
                if self.brem_flag:
                    brem_photon = 0
                    
                    brem_can_func = interp.interp1d(self.time_bins, self.brem_can[ei])
                    brem_cap_func = interp.interp1d(self.time_bins, self.brem_cap[ei])
                    brem_cancap_func = interp.interp1d(self.time_bins, self.brem_cancap[ei])
                    
                    if type(c_var) in (float, int):
                        Cap, Can = Ncouplings(gagg, gagg * 2 * np.pi * 137 / (1 + c_var))
                    elif type(c_var) in (tuple, list, np.ndarray):
                        Cap, Can = c_var
                    else:
                        print('bad coupling variable - exit')
                        exit()
                        
                    brem_photon += Cap**2 * integrator(brem_cap_func, [start_T, end_T])
                    brem_photon += Can**2 * integrator(brem_can_func, [start_T, end_T])
                    brem_photon += Cap * Can * integrator(brem_cancap_func, [start_T, end_T])
                    spec += brem_photon * (gagg / 1e-12)**2
                    
                if self.pion_flag:
                    pion_photon = 0
                    
                    pion_can_func = interp.interp1d(self.time_bins, self.pion_can[ei])
                    pion_cap_func = interp.interp1d(self.time_bins, self.pion_cap[ei])
                    pion_cancap_func = interp.interp1d(self.time_bins, self.pion_cancap[ei])
                    
                    if type(c_var) in (float, int):
                        Cap, Can = Ncouplings(gagg, gagg * 2 * np.pi * 137 / (1 + c_var))
                    elif type(c_var) in (tuple, list, np.ndarray):
                        Cap, Can = c_var
                    else:
                        print('bad coupling variable - exit')
                        exit()
                        
                    pion_photon += Cap**2 * integrator(pion_cap_func, [start_T, end_T])
                    pion_photon += Can**2 * integrator(pion_can_func, [start_T, end_T])
                    pion_photon += Can * Cap * integrator(pion_cancap_func, [start_T, end_T])
                    spec += pion_photon * (gagg / 1e-12)**2
                    
                model[ei, toi[ti]] = (Aeff[ei] / (4 * np.pi * d**2)) * spec
                
        return model
    
    
    

class analysis:
    def __init__(self, data_time, data_counts, sig_model, bkg_model, c_var = None):
        self.data_time = data_time
        self.data_counts = data_counts
        
        self.sig_model = sig_model
        self.bkg_model = bkg_model

        self.c_var = c_var
        t_nu = 27341.37
        
        self.bkg_bin = np.where(data_time < t_nu)[0]
        self.sig_bin = np.where(data_time > t_nu)[0]
        
        
        
    def negLL(self, x):
        bkg_ = self.bkg_model(x[:6])
        sig_ = x[6] * self.sig_model / np.amax(self.sig_model)
        
        LL = np.sum(xlogy(self.data_counts, (bkg_ + sig_)) - (bkg_ + sig_))
#         bkg_LL = np.sum(xlogy(self.data_counts[:,self.bkg_bin], bkg_[:,self.bkg_bin]) - bkg_[:,self.bkg_bin])
#         sig_LL = np.sum(xlogy(self.data_counts[:,self.sig_bin], (bkg_ + sig_)[:,self.sig_bin]) - (bkg_ + sig_)[:,self.sig_bin])
        
#         return np.nan_to_num(-2 * (bkg_LL + sig_LL), nan = 1e10)
        return np.nan_to_num(-2 * LL, nan = 1e10)



    def negLL_gagg(self, x):
        bkg_ = self.bkg_model(x[:6])
        sig_ = self.sig_model

        LL = np.sum(xlogy(self.data_counts, (bkg_ + sig_)) - (bkg_ + sig_))
        return np.nan_to_num(-2 * LL, nan = 1e10)