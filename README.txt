EELS Monte Carlo Simulation Framework
Author: H. Lee
Date: 2025

-------------------------------------------
Description

This repository includes two main files for simulating electron transport and EELS (Electron Energy Loss Spectroscopy):

1. MC_Hoon_EELS.py
   - Core Monte Carlo engine.
   - Contains processData() and run_MC() functions.
   - Models elastic, inelastic, and capture (DEA/EAD) processes.
   - Uses spline interpolation for energy-dependent cross sections.
   - Calculates mean free paths, energy loss distributions, and reflection/transmission effects.

2. LEE_Hoon_EELS_Multiple.ipynb
   - Jupyter notebook interface.
   - Runs multiple EELS simulations and visualizes spectra.
   - Used for comparison with experimental or thesis data.

-------------------------------------------
Key Features

- Monte Carlo electron transport in condensed-phase media.
- Includes elastic, inelastic, DEA, and EAD channels.
- Energy loss spectra (EELS) reproduction for 14.3 and 19 eV electrons.
- Computes mean free paths, excitation/ionization yields, and secondary electron distributions.

-------------------------------------------
Requirements

Python 3.9 or later
numpy, pandas, scipy, matplotlib

Data directories required:
scat_data/
Excitation/
IonDist/
ELF/
Figures/

-------------------------------------------
Usage

1. Prepare directory structure with all referenced CSV/TXT data.
2. Run from command line:
   python MC_Hoon_EELS.py
3. Or use interactively in Python:

   from MC_Hoon_EELS import processData, run_MC
   params = processData(ntimesTRAP=1, ntimesDEA=1, scat_fact=1.0)
   result = run_MC(energy=19, t_max=1e3, water_number_density=3.33e22, aniso_fact=1.0, *params)

4. For visualization and parameter sweeps:
   Open LEE_Hoon_EELS_Multiple.ipynb in Jupyter Notebook.

-------------------------------------------
Scientific Note

The model distinguishes between EELS spectrum reproduction and electron transport prediction.
Results from this work show improved accuracy in reproducing zero-loss peaks and low-energy (rotational/vibrational) losses (<1 eV).

-------------------------------------------
Citation

H. Lee et al., “Towards a Comprehensive Understanding of Low-Energy Electron Energy Loss Spectra of Amorphous Ice,” 2025 (under review).

-------------------------------------------
License

For academic and non-commercial research use only.
Please contact the author for redistribution or adaptation.
