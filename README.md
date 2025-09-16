# SN Axion Limit Check

Various checks on the SN axion emission and resulting limits.

In the notebook Analysis.ipynb, we show three cases of obtaining the 95% upper limit. First, we show the estimate of the upper limit on $g_{a \gamma \gamma}$ from Primakoff emission of our fiducial SN model without time binning information. Then, we show our full analysis in obtaining this upper limit for our fiducial analysis, which includes time binning, profiling over backgrounds, and using all energy bins. Finally, we show the resulting upper limit when bremsstrahlung axion emission is also factored into the spectra, resulting in the upper limit that appears as the solid blue line of Fig. 1 in 2405.19393. 

In the notebook AxionSpectrum.ipynb, we compare the axion emission spectrum from the Primakoff production mechanism as calculated in [2405.19393](https://arxiv.org/abs/2405.19393) to the digitized data of hot SN model and cold SN model. The peak of the spectrum seem to differ by a factor of ~1.5 when comparing hot SN model with SFHo-20.0 and the cold SN model with SFHo-18.8.

In the notebook check_Fermi.ipynb, we make an independent check on the axion constraint projection of a Fermi-LAT-like telescope observing a future 10kpc BSG. We explicitly check the KSVZ-like axion case in 2405.19393, and reproduce the constraint shown in Fig. 1 of 2405.19393.

`smm_analysis.py` now has a modified `sig_model` function under the `model` class. It now creates the time bin edges from the bin center given in the SMM data, and the function integrates between bin edges. The first and last bin in the 10s interval are included in the signal model regardless of whether the full bin is within the 10s interval of the axion emission time.
