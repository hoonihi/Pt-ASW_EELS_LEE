import math
import random
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from scipy               import interpolate, stats
from scipy.interpolate   import splev, splrep
from scipy.integrate     import quad
from scipy.stats         import beta
from scipy.optimize      import curve_fit


def processData(ntimesTRAP, ntimesDEA, scat_fact, Gcomp = False):
    
    #ALL C-S UNIT MUST BE IN 10^-16cm^2 FOR CONSISTENCY
    
    #sanche             = pd.read_csv("scat_data/sanche_signorell.csv")
    sanche             = pd.read_csv("scat_data/sanche_signorell_hoon.csv")        #10^-16cm^2 (=10^-2nm^2), Shift E0 1.0eV, set between 0.1-10keV
    energy_grid        = sanche["E0"].values                                       #2.7-10939 eV (1eV shift applied)
    
    vT                 = scat_fact * sanche["Translational1"].values
    vT_gamma           = sanche["Translational1_Anisotropy"].values
    vT_mean_loss       = 10e-3
    vT_FWHM            = 1e-3
    vT_int_spl         = splrep(energy_grid, (vT + 1e-14))
    vT_int             = lambda energy: splev(energy, vT_int_spl)
    vT_gint_spl        = splrep(energy_grid, vT_gamma, k = 1)
    vT_gint            = lambda energy: splev(energy, vT_gint_spl)
    
    vT2                = scat_fact * sanche["Translational2"].values
    vT2_gamma          = sanche["Translational2_Anisotropy"].values
    vT2_mean_loss      = 24e-3
    vT2_FWHM           = 25e-3
    vT2_int_spl        = splrep(energy_grid, (vT2 + 1e-14))
    vT2_int            = lambda energy: splev(energy, vT2_int_spl)
    vT2_gint_spl       = splrep(energy_grid, vT2_gamma, k = 1)
    vT2_gint           = lambda energy: splev(energy, vT2_gint_spl)
    
    vL                 = scat_fact * sanche["Librational1"].values
    vL_gamma           = sanche["Librational1_Anisotropy"].values
    vL_mean_loss       = 61e-3
    vL_FWHM            = 30e-3
    vL_int_spl         = splrep(energy_grid, (vL + 1e-14))
    vL_int             = lambda energy: splev(energy, vL_int_spl)
    vL_gint_spl        = splrep(energy_grid, vL_gamma, k = 1)
    vL_gint            = lambda energy: splev(energy, vL_gint_spl)
    
    vL2                = scat_fact * sanche["Librational2"].values
    vL2_gamma          = sanche["Librational2_Anisotropy"].values
    vL2_mean_loss      = 92e-3
    vL2_FWHM           = 40e-3
    vL2_int_spl        = splrep(energy_grid, (vL2 + 1e-14))
    vL2_int            = lambda energy: splev(energy, vL2_int_spl)
    vL2_gint_spl       = splrep(energy_grid, vL2_gamma, k = 1)
    vL2_gint           = lambda energy: splev(energy, vL2_gint_spl)
    
    v2                 = scat_fact * sanche["Bending"].values
    v2_gamma           = sanche["Bending_Anisotropy"].values
    v2_mean_loss       = 204e-3
    v2_FWHM            = 16e-3
    #v2[0]  = 0.025                                                             #FOR TEST (Duplicate Kai Fig. 6)
    #energy_grid[0] = 1.0                                                       
    v2_int_spl         = splrep(energy_grid, (v2 + 1e-14))
    v2_int             = lambda energy: splev(energy, v2_int_spl)
    v2_gint_spl        = splrep(energy_grid, v2_gamma, k = 1)
    v2_gint            = lambda energy: splev(energy, v2_gint_spl)
    
    v13                = scat_fact * sanche["Stretching"].values
    v13_gamma          = sanche["Stretching_Anisotropy"].values
    v13_mean_loss      = 417e-3
    v13_FWHM           = 50e-3
    #v13[0] = 0.085                                                             #FOR TEST (Duplicate Kai Fig. 6)
    #energy_grid[0] = 1.7                                                       
    v13_int_spl        = splrep(energy_grid, (v13 + 1e-14))
    v13_int            = lambda energy: splev(energy, v13_int_spl)
    v13_gint_spl       = splrep(energy_grid, v13_gamma, k = 1)
    v13_gint           = lambda energy: splev(energy, v13_gint_spl) * (energy > 7.2)
    
    v3                 = scat_fact * sanche["AsymStretch"].values
    v3_gamma           = sanche["AsymStretch_Anisotropy"].values
    v3_mean_loss       = 460e-3
    v3_FWHM            = 5e-3            
    v3_int_spl         = splrep(energy_grid, (v3 + 1e-14))
    v3_int             = lambda energy: splev(energy, v3_int_spl)
    v3_gint_spl        = splrep(energy_grid, v3_gamma, k = 1)
    v3_gint            = lambda energy: splev(energy, v3_gint_spl)
    
    v13L               = scat_fact * sanche["StretchLib"].values
    v13L_gamma         = sanche["StretchLib_Anisotropy"].values
    v13L_mean_loss     = 500e-3
    v13L_FWHM          = 40e-3
    v13L_int_spl       = splrep(energy_grid, (v13L + 1e-14))
    v13L_int           = lambda energy: splev(energy, v13L_int_spl)
    v13L_gint_spl      = splrep(energy_grid, v13L_gamma, k = 1)
    v13L_gint          = lambda energy: splev(energy, v13L_gint_spl)
    
    v213               = scat_fact * sanche["OvertoneStretch"].values
    v213_gamma         = sanche["OvertoneStretch_Anisotropy"].values
    v213_mean_loss     = 835e-3
    v213_FWHM          = 75e-3
    v213_int_spl       = splrep(energy_grid, (v213 + 1e-14))
    v213_int           = lambda energy: splev(energy, v213_int_spl)
    v213_gint_spl      = splrep(energy_grid, v213_gamma, k = 1)
    v213_gint          = lambda energy: splev(energy, v213_gint_spl)

    #Energy loss and gamma vectors
    losses             = np.array([vT_mean_loss,  vT2_mean_loss,   vL_mean_loss, vL2_mean_loss, v2_mean_loss, v13_mean_loss, \
                                   v3_mean_loss, v13L_mean_loss, v213_mean_loss])
    gammas             = np.array([vT_gint, vT2_gint, vL_gint, vL2_gint, v2_gint, v13_gint, v3_gint, v13L_gint, v213_gint])
    """
    losses_FWHM        = np.array([vT_FWHM, vT2_FWHM, vL_FWHM, vL2_FWHM, v2_FWHM, v13_FWHM, v3_FWHM, v13L_FWHM, v213_FWHM])
    other_losses       = np.array([7.5, 10.4, 10.9, 13.5, 14.5, 17, 28])                               #8.5 in Sanche paper
    other_excitation   = np.array([  1,    1,    0,    0,    1,  0,  1])      
    coll_names         = ["vT", "vT2", "vL", "vL2", "v2", "v13", "v3", "v13L", "v213", "other"]
    """
    
    #Inelastic C-S (Inferior: Rot. Vib. Lib.): Michaud & Sanche 2003
    inferior           = v213 + v13L + v3 + v13 + v2 + vL2 + vL + vT2 + vT
    inferior_spl       = splrep(energy_grid, inferior)
    inferior_int       = lambda energy: splev(energy, inferior_spl)

    #Inelastic C-S (Exc., Ion.): Michaud & Sanche 2003 ------> w/o DEA hump at 4-6eV (removed from original data)
    others             = pd.read_csv("scat_data/sanche_others_2003_noDEA.csv")              #10^-16 cm^2
    energy_grid_others = others["E0"].values                                                #1.0eV shifted
    data_others        = scat_fact * others["Others"].values
    others_spl         = splrep(energy_grid_others, data_others)  
    others_int         = lambda energy: splev(energy, others_spl, ext=1)
    
    #Inelastic C-S (Total) and vectors
    inelastic_int      = lambda energy: (inferior_int(energy) + others_int(energy))*1
    inel_vec           = lambda energy: np.abs([0, vT_int(energy)*1,vT2_int(energy)*1,  vL_int(energy)*1, vL2_int(energy)*1,    v2_int(energy)*1, \
                                                  v13_int(energy)*1, v3_int(energy)*1,v13L_int(energy)*1,v213_int(energy)*1,others_int(energy)*1])

    #Excitation ejection length (Elles, 2006)
    exc_el_Elles       = pd.read_csv("Excitation/Elles_EjectionLength.csv")
    exc_el_grid        = exc_el_Elles["E0"].values
    exc_el             = exc_el_Elles["EL"].values
    exc_el_spl         = splrep(exc_el_grid, exc_el)
    exc_el_int         = lambda energy: splev(energy, exc_el_spl)
    
    #Excitation survival probability (Elles, 2006)
    exc_sp_Elles       = pd.read_csv("Excitation/Elles_SurvivalProbability.csv")
    exc_sp_grid        = exc_sp_Elles["E0"].values
    exc_sp             = exc_sp_Elles["SP"].values
    exc_sp_spl         = splrep(exc_sp_grid, exc_sp)
    exc_sp_int         = lambda energy: splev(energy, exc_sp_spl, ext=3) * (energy >= 6)
    
    #Photoionization efficiency (Mozumder, 2002)
    Photo_eff_Mozumder = pd.read_csv("Excitation/Mozumder_PhotoEff.csv")
    Photo_eff_Mgrid    = Photo_eff_Mozumder["E0"].values
    Photo_eff_M        = Photo_eff_Mozumder["Eff"].values
    Photo_eff_Mspl     = splrep(Photo_eff_Mgrid, Photo_eff_M)
    Photo_eff_Mint     = lambda energy: splev(energy, Photo_eff_Mspl, ext=3) * (energy >= 5)

    #Photoionization efficiency (Tan, 1978)
    Photo_eff_Tan      = pd.read_csv("Excitation/Tan_PhotoEff.csv")
    Photo_eff_Tgrid    = Photo_eff_Tan["E0"].values
    Photo_eff_T        = Photo_eff_Tan["Eff"].values
    Photo_eff_Tspl     = splrep(Photo_eff_Tgrid, Photo_eff_T)
    Photo_eff_Tint     = lambda energy: splev(energy, Photo_eff_Tspl, ext=3) * (energy >= 10)
    
    #Exc. vs. Ion. fraction (Kyriakou 2015)
    ion_frac_Kyriakou  = pd.read_csv("IonDist/ionization_fraction.csv")
    ion_frac_grid      = ion_frac_Kyriakou["E0"].values                               #10eV - 10keV
    ion_frac           = ion_frac_Kyriakou["Ionization"].values
    ion_frac_spl       = splrep(ion_frac_grid, ion_frac)
    ion_frac_int       = lambda energy: splev(energy, ion_frac_spl) * (energy >= 10)
    
    #Elastic C-S
    #elastic             = np.loadtxt("scat_data/Elastic_CS_Sanche.csv", delimiter = ",")                      #Sanche extrapolation
    #elastic             = np.loadtxt("scat_data/Elastic_CS_Pimblott.csv", delimiter = ",")                    #Pimblott 1996
    #elastic             = np.loadtxt("scat_data/Elastic_CS_Kai.csv", delimiter = ",")                         #Kai 2023 (Moliere 1948)
    #elastic             = np.loadtxt("scat_data/Elastic_CS_Worner_extrapolated_EMFP.csv", delimiter = ",")    #Worner 2022
    elastic             = np.loadtxt("scat_data/Elastic_CS_Signorell_extrapolated_hoon.csv", delimiter = ",") #Sanche ext by Signorell
    elastic_grid        = elastic[:, 0]                                                                       #1eV shift applied
    elastic_CS          = elastic[:, 1]                                                                       #10^-16cm^2 (=10^-2nm^2)
    elastic_spl         = splrep(elastic_grid, elastic_CS)
    elastic_int         = lambda energy: splev(energy, elastic_spl)      * 20         #scaling to include aniso component
    
    
    #Elastic_ADCS (Worner, 2022)
    elastic_ADCS_Worner = pd.read_csv("scat_data/Elastic_ADCS_Worner.csv")
    elastic_ADCS_grid   = elastic_ADCS_Worner["E0"].values
    elastic_ADCS        = elastic_ADCS_Worner["mu"].values
    elastic_ADCS_spl    = splrep(elastic_ADCS_grid, elastic_ADCS)
    elastic_ADCS_int    = lambda energy: splev(energy, elastic_ADCS_spl, ext=3)
    
    """
    #Sum of two C-S derived from EMFP and MTMFP respectively
    elastic_EMFP        = np.loadtxt("scat_data/Elastic_CS_Worner_extrapolated_EMFP.csv", delimiter = ",")  #Worner 2022 ext up to 10keV
    elastic_EMFP_grid   = elastic_EMFP[:, 0]
    elastic_EMFP_CS     = elastic_EMFP[:, 1]
    elastic_EMFP_spl    = splrep(elastic_EMFP_grid, elastic_EMFP_CS)
    elastic_EMFP_int    = lambda energy: splev(energy, elastic_EMFP_spl)
    
    #elastic_MTMFP       = np.loadtxt("scat_data/Elastic_CS_Worner_extrapolated_MTMFP.csv", delimiter = ",") #Worner 2022 ext up to 10keV
    elastic_MTMFP       = np.loadtxt("scat_data/Elastic_CS_Signorell_extrapolated.csv", delimiter = ",")
    elastic_MTMFP_grid  = elastic_MTMFP[:, 0]
    elastic_MTMFP_CS    = elastic_MTMFP[:, 1]
    elastic_MTMFP_spl   = splrep(elastic_MTMFP_grid, elastic_MTMFP_CS)
    elastic_MTMFP_int   = lambda energy: splev(energy, elastic_MTMFP_spl)
    
    elastic_int         = lambda energy: elastic_EMFP_int(energy) + elastic_MTMFP_int(energy)
    """
    
    #"""
    #Trapping C-S: ref. Konovalov 1988, Green & Pimblott 2001 (total C-S), Signorell 2020 (total C-S)
    mu_trap            = 0
    variance_trap      = 0.4
    sigma_trap         = math.sqrt(variance_trap)                                                                 #Standard deviation
    trap_int           = lambda energy: stats.norm.pdf(energy, mu_trap, sigma_trap) * 2.5 * ntimesTRAP #* 5.0e-2        #10^-16cm^2
    #"""
    
    #DEA C-S: ref. Melton 1972, Itikawa 2005, Rawat 2007, Song 2021 --- 3 channels (H-, O-, OH-) gas phase data
    DEA                = pd.read_csv("scat_data/DEA_CS_Song.csv")
    energy_grid_DEA    = DEA["E"].values + 0                                                               #Shift for ASW? (not applied)
    data_DEA           = DEA["DEA"].values                                                                 #10^-16cm^2 (=10^-2nm^2)
    DEA_spl            = splrep(energy_grid_DEA, data_DEA)  
    #DEA_int            = lambda energy: splev(energy, DEA_spl) * (energy>=4.5) * (energy<=19.5) * ntimesDEA
    DEA_int            = lambda energy: splev(energy, DEA_spl, ext=1) * ntimesDEA                          #Scaling (1/20?) for liquid-phase
    
    #EAD C-S
    Probability_EAD     = 85                                                                        #85% of TNA undergo EAD (Goursaud, 1976)
    EAD_int             = lambda energy: DEA_int(energy) * (Probability_EAD / (100 - Probability_EAD))

    
    #Capture C-S
    capture_int         = lambda energy: DEA_int(energy) + EAD_int(energy) #         + trap_int(energy)    #No trapping?
        
    #Total C-S    
    total_int           = lambda energy: inelastic_int(energy) + elastic_int(energy) + capture_int(energy)
        




    #DOSD (0-100eV)    
    dosd              = np.loadtxt("IonDist/DOSD_0-169.csv", delimiter = ",")
    dosd_grid         = dosd[:, 0]
    dosd_pdf          = dosd[:, 1]
    dosd_spl          = splrep(dosd_grid, dosd_pdf)
    dosd_int          = lambda energy: splev(energy, dosd_spl) * (energy < 100)
    #res1, err1        = quad(dosd_int, 0, 100, limit=1000)                                    #integration over the range: 0-100eV
    res1, err1        = quad(dosd_int, 0, 100, limit=1000, points=[20,25])
    
    #DOSD (100-540eV)
    x_ext_grid1       = np.linspace(100, 540, 540 - 100 + 1)
    m1                = 0.013734276945027433
    t1                = 0.023989038918209613
    manual_spl1       = splrep(x_ext_grid1, [m1 * np.exp(-t1 * (i-100)) for i in x_ext_grid1])
    extra_int1        = lambda energy: splev(energy, manual_spl1) * ((energy >= 100) & (energy < 540))
    res2, err2        = quad(extra_int1, 100, 540, limit=1000)
    
    #DOSD (540-10keV)
    m2                = 0.004922410639512823                         #from LEE_Hoon_DOSD_extrapolation
    t2                = 0.003608981834456128 # * 2
    nikjoo_ext_grid2  = np.linspace(540, 10000, 10000 - 540 + 1)
    extra_nikjoo_spl2 = splrep(nikjoo_ext_grid2, [m2 * np.exp(-t2 * (i-540)) for i in nikjoo_ext_grid2])
    extra_nikjoo_int2 = lambda energy: splev(energy, extra_nikjoo_spl2) * ((energy >= 540) & (energy <= 10000))
    res3, err3        = quad(extra_nikjoo_int2, 540, 10000, limit=1000)
    
    nist_grid         = nikjoo_ext_grid2
    nist_int          = extra_nikjoo_int2

    #Validation: Integration area (res) = 10?
    res = res1 + res2 + res3
    """
    print("res1 (  0-100eV):", "%.2f" % res1)
    print("res2 (100-540eV):", "%.2f" % res2)
    print("res3 (540-10keV):", "%.2f" % res3)
    print("res             :", "%.2f" % res)
    """
    
    """
#Option 1
    nist       = np.loadtxt("IonDist/NIST_540-10000.csv", delimiter = ",")
    nist_grid  = nist[:, 0]
    nist_pdf   = nist[:, 1] #*0.7
    nist_spl   = splrep(nist_grid, nist_pdf)
    nist_int   = lambda energy: splev(energy, nist_spl)
    res0, err0 = quad(nist_int, 540, 10000)
    """
    """
#Option 2
    #nikjoo2      = np.loadtxt("IonDist/Nikjoo_2009_Hydrogenic.csv", delimiter = ",")
    nikjoo2      = np.loadtxt("IonDist/Nikjoo_2009_Drude.csv", delimiter = ",")
    nikjoo_grid2 = nikjoo2[:, 0]
    nikjoo_pdf2  = nikjoo2[:, 1]  / 2 / np.pi / np.pi / 3.34 * nikjoo_grid2
    nikjoo_spl2  = splrep(nikjoo_grid2, nikjoo_pdf2)
    nikjoo_int2  = lambda energy: splev(energy, nikjoo_spl2)
    res0, err0   = quad(nikjoo_int2, 540, 10000)
    nist_grid    = nikjoo_grid2
    nist_int     = nikjoo_int2
    """
    """
#Option 3
    #nikjoo2           = np.loadtxt("IonDist/Nikjoo_2009_Hydrogenic.csv", delimiter = ",")
    nikjoo2           = np.loadtxt("IonDist/Nikjoo_2009_Drude.csv", delimiter = ",")
    nikjoo_grid2      = nikjoo2[:, 0]
    nikjoo_pdf2       = nikjoo2[:, 1] / 2 / np.pi / np.pi / 3.34 * nikjoo_grid2
    nikjoo_spl2       = splrep(nikjoo_grid2, nikjoo_pdf2)
    nikjoo_int2       = lambda energy: splev(energy, nikjoo_spl2)
    """
    """
    plt.figure(figsize=(20, 8))
    plt.xlim(0, 700)
    plt.ylim(0, 0.25)
    plt.xlabel("Energy transfer (eV)", fontsize = 16)
    plt.ylabel("DOSD (1/eV)",          fontsize = 16)
    plt.plot(dosd[0:79, 0], dosd[0:79, 1], 'ko', linewidth=3.0, label="0-100eV: DOSD data (Hayashi et al.,2000)")
    plt.plot(x_ext_grid1,   extra_int1(x_ext_grid1), 'r-', linewidth=3.0, label="100-540eV: Mono Exp. fit to (Hayashi et al.,2000) & extrapolation")
    plt.plot(nist_grid,     nist_int(nist_grid), 'b-', linewidth=3.0, label="540eV-10keV: Mono Exp. fit to (Emfietzoglou et al., 2009)")
    plt.xticks(np.arange(0, 701, 50), fontsize = 14)
    plt.yticks(                       fontsize = 14)
    plt.grid()
    plt.legend(fontsize = 18)
    #plt.savefig("Figures/DOSD.png", dpi=600)
    """
    
    #Energy Loss Function (k=0)
    ELF      = np.loadtxt("IonDist/ELF.csv", delimiter = ",")                                 #0-100eV
    ELF_grid = ELF[:, 0]
    ELF_pdf  = ELF[:, 1]
    ELF_spl  = splrep(ELF_grid, ELF_pdf)
    ELF_int  = lambda energy: splev(energy, ELF_spl, ext=1) * (energy < 100)
    
    elf_grid = np.linspace(100, 10000, 10000 - 100 + 1)                                       #100-10keV (fit from 50eV))
    m_elf    = 0.1428419081607864 
    t_elf    = 0.04895059734865324
    elf_spl  = splrep(elf_grid, [m_elf * np.exp(-t_elf * (i-50)) for i in elf_grid])
    elf_int  = lambda energy: splev(energy, elf_spl) * ((energy >= 100) & (energy <= 10000))
        
    ELOSS_int = lambda energy: ELF_int(energy) + elf_int(energy)

    
    #Density of State (DOS) for sampling Q_ion (i.e. binding energies)
    DOS_grid, cumul_dos = np.loadtxt("IonDist/ValenceDOS_low_2a1.txt")
    DOS_pdf             = np.zeros(len(DOS_grid))
    DOS_pdf[1:]         = cumul_dos[1:] - cumul_dos[:-1].copy()
    DOS_spl             = splrep(DOS_grid[1:], DOS_pdf[1:])
    DOS_int             = lambda energy: splev(-energy, DOS_spl, ext=1)

    #Excitation energy Loss distribution
    EXC_loss      = np.loadtxt("Excitation/Excitation_Eloss.csv", delimiter = ",")
    EXC_loss_grid = EXC_loss[:, 0]
    EXC_loss_pdf  = EXC_loss[:, 1]
    EXC_loss_spl  = splrep(EXC_loss_grid, EXC_loss_pdf)
    EXC_loss_int  = lambda energy: splev(energy, EXC_loss_spl, ext=1)

    Pt_transmitted  = np.loadtxt("scat_data/Pt_Transmitted_Bader.csv", delimiter = ",")
    Pt_reflect_grid = Pt_transmitted[:, 0]
    Pt_reflect_pdf  = (3 - Pt_transmitted[:, 1]) / 3                          #Reflected current = Incident current(3*E-9) - transmitted current
    Pt_reflect_spl  = splrep(Pt_reflect_grid, Pt_reflect_pdf)
    Pt_reflect_int  = lambda energy: splev(energy, Pt_reflect_spl, ext=1)     #Reflectivity = Reflected current / Incident current
    
    print("Inelastic (Vib) MFP: " + "{:.2f}".format(1/inelastic_int(14.3)/3.125e22*1e16*1e7) + " nm (For 14.3 eV)")
    print("Elastic MFP        : " + "{:.2f}".format(1/  elastic_int(14.3)/3.125e22*1e16*1e7) + " nm (For 14.3 eV)")
    
    return (total_int, capture_int, inelastic_int, inel_vec, ion_frac_int, losses, gammas, elastic_int, others_int, EAD_int, trap_int, \
            DEA_int, vT_int, vT2_int, vL_int, vL2_int, v2_int, v13_int, v213_int, v13L_int, v3_int, vT_gint, vT2_gint, vL_gint,        \
            vL2_gint, v2_gint, v13_gint, v213_gint, v13L_gint, v3_gint, inferior_int, res, res1, res2, res3, dosd_int, extra_int1,     \
            nist_int, ELOSS_int, DOS_int, EXC_loss_int, exc_el_int, exc_sp_int, Photo_eff_Mint, Photo_eff_Tint, elastic_ADCS_int,      \
            Pt_reflect_int)



def run_MC(energy, t_max, water_number_density, aniso_fact, capture_int, inelastic_int, exc_el_int, exc_sp_int, Photo_eff_Mint,        \
           Photo_eff_Tint, total_int, inel_vec, losses, gammas, ion_frac_int, energy_shift, e_cutoff, res, res1, res2, res3, dosd_int, \
           extra_int1, nist_int, ELOSS_int, DOS_int, EXC_loss_int, others_int, thickness, energy_ini, elastic_ADCS_int, Pt_reflect_int):

    gamma               = 0.0                                                    #azimuthal angle
    mass_e              = 9.10938356e-31                                         #kg
    eV2J                = 1.60217646e-19                                         #Joule/eV
    #density_factor      = water_number_density * 1e-21                           #adjusts for the units of the cross-sections: /cm3 to /nm3
    tot_distance        = 0.0
    tot_t_elapsed       = 0.0
    depth               = []
    dist_travel         = []
    t_travel            = []
    meanDOSD            = []
    energy_evol         = []
    pos_evol            = []
    pos_evol_specific   = []
    ion_init            = []                                                     #to keep track of ionization losses for debugging
    ion_loss            = []
    ion_pair            = []
    ifcaptured          = []
    energy_incident     = []
    e_aq_Gval_Ion       = [0]
    e_aq_Gval_Exc_EAD   = [0]
    e_aq_Gval_TNA_EAD   = [0]   
    
    eloss               = []
    eloss_inelastic_per_sample = []
    energy_second       = []
    t_ionization        = []
    t_excitation        = []
    t_TNA_EAD           = []
    
    Q_data              = []
    cosine_data         = []
    MT_data             = []
    E_ion_data          = []
    E_ion_loss_data     = []
    E_exc_loss_data     = []
    E_inel_loss_data    = []
    T_d_data            = []
    
    E_loss_EELS_back              = []
    E_loss_EELS_back_specific     = []
    E_loss_EELS_back_specific_ion = []
    E_loss_EELS_back_specific_exc = []
    
    energy_ead          = []
    energy_ionpair      = []
    energy_loss         = []
    secondary           = []
    EELS_second_energy  = []
    
    inelastic_coll      = 0
    elastic_coll        = 0
    excitation_coll     = 0
    excitation_EAD_coll = 0
    excitation_EAD      = []
    TNA_EAD             = []
    ionization_coll     = 0
    #TNA_coll            = 0
    DEA_coll            = 0
    TNA_EAD_coll        = 0
    trap_coll           = 0
    cap_coll            = 0
    DOSD_coll           = 0
    EELS_back_coll      = 0
    EELS_through_coll   = 0
    surface_reflect_coll= 0
    EELS_second_coll    = 0
    
    therm_num           = 0 
    count_all           = 0
    count_cap           = 0

    
    
    if energy < e_cutoff:
        therm_num=1
    
    pos = np.array([0, 0, 0], dtype=float)
    last_condition = None
    density_factor = water_number_density * 1e-21                              #3.3367e22, adjusts for the units:: /cm3 to /nm3
    
    
    
    #"""
    ########################### EELS ###########################    
    first_iteration = True
    if energy_ini == 14.3 or energy_ini == 19:                                 #Initial direction cosine (=0.97) for 14.3 and 19 eV cases
        mu        = np.cos(np.radians(14))                                            
        detect_i  = 119
        detect_f  = 123
    else:                                                                      #100 eV case
        mu        = np.cos(np.radians(40))
        detect_i  = 41 #135
        detect_f  = 45 #1391
    
    detect_min = min(np.cos(np.radians(detect_i)), np.cos(np.radians(detect_f)))
    detect_max = max(np.cos(np.radians(detect_i)), np.cos(np.radians(detect_f)))
    
    while (energy >= e_cutoff and tot_t_elapsed < t_max and 0 <= pos[2] <= thickness):            #Multiple scattering within film thickness
    ############################################################
    #"""
                
        energy      = energy + energy_shift            
        tot_val     = density_factor *     total_int(energy) * 1e-2            #Macroscopic cross-section (1/nm)
        inl_val     = density_factor * inelastic_int(energy) * 1e-2            #Inferior + Others (Exc, Ion)
        cap_val     = density_factor *   capture_int(energy) * 1e-2            #Trapping + DEA + EAD
                
        r_dist      = random.random()
        distance    = -math.log(1 - r_dist) / tot_val                          #distance (MFP) to move
        speed       = math.sqrt(2*energy*eV2J / mass_e) * 1e9 * 1e-15          #nm/fs
        t_elapsed   = distance / speed                                         #fs  
        
        if (tot_t_elapsed + t_elapsed) >= t_max:                               #linear int for the cases exceeds t_max in the last iteration
            distance_new = distance / (t_elapsed / (t_max - tot_t_elapsed))
            t_elapsed    = t_elapsed / (distance / distance_new)
            distance     = distance_new
        tot_t_elapsed += t_elapsed              
        tot_distance  += distance  

        r_type      = random.random()
        
        
        ################################### EELS SIMULATION (Consider reflection and transmission) ###################################
        #"""        

        
#ASW surface reflection ?
        if energy_ini == 14.3:                       #14.3 eV case 
            ASW_reflect_threshold = 0.25
        elif energy_ini == 19:                       #19 eV case
            ASW_reflect_threshold = 0.22
        else:                                        #100 eV case
            ASW_reflect_threshold = 0
        r_ASW_reflectivity = random.random()
        if first_iteration and r_ASW_reflectivity <= ASW_reflect_threshold:              #direct surface refelection
            gamma = random.uniform(0, 2*math.pi)
            mu    = random.uniform(np.cos(np.radians(180)), np.cos(np.radians(90)))      #Iso scattering due to the surface roughness (-1~0)
            pos  += [distance * math.sqrt(1-mu**2) * math.cos(gamma), \
                     distance * math.sqrt(1-mu**2) * math.sin(gamma), \
                     distance * mu]                                       #pos[2] becomes < 0 ---> Handled at line 566
        else:
            first_iteration = False        

#Pt surface reflection ?
        if (pos[2] + distance*mu) > thickness:                            #reflection from Pt substrate in EELS simulation
            if energy_ini == 14.3:                   #14.3 eV case 
                Pt_reflect_threshold = Pt_reflect_int(energy)#0.56
            elif energy_ini == 19:                   #19 eV case
                Pt_reflect_threshold = Pt_reflect_int(energy)#0.50
            else:                                    #100 eV case
                Pt_reflect_threshold = Pt_reflect_int(energy)#0
            r_Pt_reflectivity = random.random()           
            if r_Pt_reflectivity <= Pt_reflect_threshold:                 #Michaud,1987 (0.15 for 8eV, comparable to semi-infinite layers for 20eV)
                pos[2] = thickness                                        #collision at the surface ---> Handled at line 571
            else:
                gamma = random.uniform(0, 2*math.pi)
                pos += [distance * math.sqrt(1-mu**2) * math.cos(gamma), \
                        distance * math.sqrt(1-mu**2) * math.sin(gamma), \
                        distance * mu]                                    #passed through ---> Handled at line 583
        
#Bacscattered to vacuum ?
        elif (first_iteration == False) and (pos[2] + distance*mu) < 0:
            E_b_surf = 1.0#0.8                                                #electron affinity at the surface (Worner, 2022)
            E_b_bulk = 0.2                                                #electron affinity in the bulk (Worner, 2022)
            e_del    = 0#E_b_surf - E_b_bulk                                #electrons are faster by e_del at the surface
            r_theta  = random.uniform(0, math.pi)                         #approaching angle (rad)
            if E_b_surf < ((energy+e_del) * (np.cos(r_theta))**2):        #Transmissino probability
                T_d = 4 *      np.sqrt(1 - (E_b_surf / ((energy+e_del) * (np.cos(r_theta))**2)))        \
                      /   (1 + np.sqrt(1 - (E_b_surf / ((energy+e_del) * (np.cos(r_theta))**2))))**2
            else:
                T_d = 0            
            
            r_prime_escape = random.random()                
            if r_prime_escape < T_d:                                  #T_d% escaped (=backscattered)
                gamma = random.uniform(0, 2*math.pi)
                pos += [distance * math.sqrt(1-mu**2) * math.cos(gamma), \
                        distance * math.sqrt(1-mu**2) * math.sin(gamma), \
                        distance * mu]                                #pos[2] becomes < 0 ---> Handled at line 586
            else:                   
                #print(energy, T_d)
                pos[2] = 0                                            #back to the film  ---> Handled at line 596
     
#Stay in the film
        else:
            gamma = random.uniform(0, 2*math.pi)                          #can be set to 0 for visualization purpose
            pos += [distance * math.sqrt(1-mu**2) * math.cos(gamma), \
                    distance * math.sqrt(1-mu**2) * math.sin(gamma), \
                    distance * mu]                                        #stay within the film ---> GOTO line 616

        ######################################################################################################################################
                    
        if first_iteration and pos[2] < 0:                                          #ASW surface reflection (Treat like an elastic scattering)
            surface_reflect_coll += 1
            eloss.append(0)
            first_iteration       = False            
        
        elif (energy_ini == 14.3 or energy_ini == 19) and (pos[2] == thickness):    #Pt surface reflection (Treat like an elastic scattering)
            eloss.append(0)
            #gamma  = random.uniform(0, 2*math.pi)
            gamma = 0
            #mus    = random.uniform(np.cos(np.pi), np.cos(np.pi/2))                                  #isotropic (Michaud, 1987) --> questionable for metal
            if energy_ini == 14.3:
                mus = np.cos(np.pi - 2*np.arccos(mu)) 
                #mus = random.uniform(np.cos(np.pi), np.cos(np.pi/2 - np.arccos(mu)))                  #restricted isotropic
            elif energy_ini == 19:
                mus = np.cos(np.pi - 2*np.arccos(mu))                                                 #specular --> ideal for Pt
            mu     = mus*mu - math.sqrt(1-mus**2) * math.sqrt(1-mu**2) * math.cos(gamma)
                        
        elif (energy_ini == 14.3 or energy_ini == 19) and (pos[2] > thickness):
            EELS_through_coll += 1
        
        elif (energy_ini == 14.3 or energy_ini == 19) and (pos[2] < 0):
            EELS_back_coll += 1
            E_loss_EELS_back.append(energy_ini-energy)                    #all direction                
            if detect_min < mu < detect_max:                              #for specific angle: +-2 deg (cf. cos(121 deg) = -0.515)
                E_loss_EELS_back_specific.append(energy_ini-energy)
                if last_condition == "ionization":
                    E_loss_EELS_back_specific_ion.append(eloss_ion)
                elif last_condition == "excitation":
                    E_loss_EELS_back_specific_exc.append(eloss_exc)
        
        elif (energy_ini == 14.3 or energy_ini == 19) and (pos[2] == 0):  #failed to escape (Treat like an elastic scattering)
            eloss.append(0)
            gamma  = random.uniform(0, 2*math.pi)
            #mus    = random.uniform(0, 1)                                #isotropic assumption
            mus    = np.cos(np.pi - 2*np.arccos(mu))                      #specular direction
            mu     = mus*mu - math.sqrt(1-mus**2) * math.sqrt(1-mu**2) * math.cos(gamma)
            
        elif (energy_ini == 100) and (pos[2] > thickness):
            EELS_back_coll += 1
            E_loss_EELS_back.append(energy_ini-energy)                    #all direction                        
            detect_min=-1
            detect_max=1
            if detect_min <= mu <= detect_max:                            #for specific angle: +-2 deg (cf. cos(121 deg) = -0.515)
                E_loss_EELS_back_specific.append(energy_ini-energy)
                if last_condition == "ionization":
                    E_loss_EELS_back_specific_ion.append(eloss_ion)
                elif last_condition == "excitation":
                    E_loss_EELS_back_specific_exc.append(eloss_exc)

                    
        elif (r_type < inl_val / tot_val):                                   #INELASTIC
        #"""
        ############################################################################################################################
        
 
        
            inels = inel_vec(energy)                                           #Array of 11: figure out what kind of inferior
            inels = inels / inels.sum()                                        #normalize
            cs    = np.cumsum(inels)                                           #Array of 11, from 0 to 1

            r_inel_type = random.random()                                      #of the types of inelastic collisions, what did we select
            for i in range(1, inels.size):                                     #range(1, 11) --> iterate 1~10
                if (r_inel_type >= cs[i - 1]) and (r_inel_type < cs[i]):       #cs[0]<=r_inel_type<cs[1], ..., cs[9]<=r_inel_type<cs[10]
                    selected = i - 1                                           #0~9: Inferior(0~8), Others-DEA(9)

            if (selected < 9):                                                 #selected = 0~8 --> INFERIOR
                inelastic_coll += 1                                                               
                eloss_inel = losses[selected]                
                gamma      = random.uniform(0, 2 * math.pi)                    #azimuthal angle change = iso (OpenMC manual)

                r_iso      = random.random()
                if (r_iso >= gammas[selected](energy) * aniso_fact):           #iso.scattering (only change directions)
                       mu  = random.uniform(-1, 1)                             #scattering cosine = omega(w) in paper
                else:                                                          #aniso scattering --> Apply kinematics
                    A      = 1                                                 #neutron-hydrogen collision
                    Q_inel = 0                                                 #negligible
                    mus    = 0.5*((A+1)*math.sqrt((energy-eloss_inel)/energy)-Q_inel*A/math.sqrt(energy*(energy-eloss_inel)))      #Shultis and Faw
                    mu     = mus*mu - math.sqrt(1-mus**2) * math.sqrt(1-mu**2) * math.cos(gamma)      #New z-direction cosine (w' in OpenMC manual)

                eloss_inelastic_per_sample.append(eloss_inel)
                eloss.append(eloss_inel)
                #if (mu > -0.52) & (mu < -0.48):
                #    eloss.append(eloss_inel)

                energy     -= eloss_inel
                E_inel_loss = eloss_inel
                E_inel_loss_data.append(E_inel_loss)
                
                last_condition = "inferior"

            else:                                                              #selected = 9 --> EXC. + ION. (excl. DEA)
                DOSD_coll       += 1                                           #Apply DOSD to both ionization and excitation
                
                if energy > 100:

                    ######################## Forward approach (Enrgy loss from ELF (k=0) ######################

                    #"""
                    if (energy < 10):                                                       #Define ionization efficiency
                        ioneff = 0                                                 
                    else:                                                                   #0.391 for 19 eV, 0.168 for 14.3 eV (Kyriakou, 2015)
                        ioneff = ion_frac_int(energy)
                    #"""
                    """
                    #For EELS data comparison
                    if energy == 14.3:
                        ioneff=0.13
                    elif energy == 19:
                        ioneff=0.44
                    """

                    r_ion_exc = random.random()

                    if (r_ion_exc <= ioneff):                                      ##### IONIZATION #####
                        ionization_coll += 1

                        if (energy <= 40):                                             #Define and sample ion. threshold, Q_ion (=binding E)
                            interval_Q = np.linspace(10, energy*8/9, 100)              #Q_max = (8/9)*E 
                        else:
                            interval_Q = np.linspace(10, 40*8/9, 100)
                        Q_cumsum = np.cumsum(DOS_int(interval_Q))

                        if Q_cumsum[-1] == 0:
                            Q_cdf = np.zeros(len(Q_cumsum))
                        else:
                            Q_cdf = Q_cumsum / Q_cumsum[-1]

                        r_DOS    = random.random()
                        Q_ion    = np.interp(r_DOS, Q_cdf, interval_Q)

                        A             = 1                                              #Neutron kinematics (A=1: neutron scattering from H)
                        cos_theta     = -1                                             #e_prime becomes minimum when cosine=-1
                        e_prime_min   = (1/(A+1)**2) * (np.sqrt(energy) * cos_theta + \
                                        np.sqrt(energy  * (A**2-1+cos_theta**2)+A*(A+1)*Q_ion))**2  #Shultis and Faw Eq(6.25)
                        e_loss_max    = energy - e_prime_min

                        if energy < 1500:                                              #ELF (k=0) is low enough (6.7E-6) at 1,500eV
                            interval  = np.linspace(Q_ion, e_loss_max, 100)
                        else:                                                          #e_loss_max is high enough (999.4) at 1,000eV
                            interval  = np.linspace(Q_ion, 1500, 100)

                        eloss_cumsum  = np.cumsum(ELOSS_int(interval))
                        eloss_cdf     = eloss_cumsum / eloss_cumsum[-1]
                        r_ion         = random.random()
                        eloss_ion     = np.interp(r_ion, eloss_cdf, interval)               

                        gamma         = random.uniform(0, 2 * math.pi)                 #azimuthal angle
                        mus           = 0.5*((A+1)*np.sqrt((energy-eloss_ion)/energy)-Q_ion*A/np.sqrt(energy*(energy-eloss_ion)))
                        mu            = mus*mu-np.sqrt(1-mus**2)*np.sqrt(1-mu**2)*math.cos(gamma)   #new z-direction cosine (w' in OpenMC)
                        
                        #delete?
                        if (energy < eloss_ion) or (mus**2 > 1):
                            print(energy, Q_ion, e_loss_max, eloss_ion, mus)

                        
                        Q                = Q_ion                                       #for G Value calculation
                        cosine           = mus
                        E_ion            = energy
                        E_ion_loss       = eloss_ion * 100 / energy                    #%
                        MT               = abs(2*np.pi/(6.626E-34/(np.sqrt(2*9.11E-31*energy*1.6E-19))*18897259886)*np.sqrt(2-2*mus))

                        Q_data.append(Q)
                        cosine_data.append(cosine)
                        E_ion_data.append(E_ion)
                        E_ion_loss_data.append(eloss_ion)
                        MT_data.append(MT)
                        
                        energy -= eloss_ion                                            #track primary electron (eloss_exc or eloss_ion)
                        energy_secondary = eloss_ion - Q_ion
                        
                        #DOSDloss.append(eloss_ion)
                        t_ionization.append(tot_t_elapsed)
                        eloss.append(eloss_ion)
                        energy_second.append(energy_secondary)
                        
                        last_condition = "ionization"
                        

                    else:                                                          ##### EXCITATION #####
                        excitation_coll     += 1                    
                        excitation_EAD_coll += int(exc_sp_int(energy) * Photo_eff_Tint(energy))  #Don't update depth here. Electrons are ADDED &
                                                                                       #track NOT terminated in case of Exc-EAD.

                        #interval_exc = np.linspace(0, energy-0.8)                     #eloss between E-1 and E won't be detected
                        if energy < 1000:                                              #ELF is low enough (6E-5) at 1,000eV
                            interval_exc  = np.linspace(7.4, energy-0.8, 100)
                        else:                                                          #e_loss_max is high enough (999.4) at 1,000eV
                            interval_exc  = np.linspace(7.4, 1000, 100)


                        gamma         = random.uniform(0, 2*math.pi)
                        """                                                            #Angular deflection assumption
                        #Lower limit: Isotropic scattering (like elastic collision)
                        mu            = random.uniform(-1, 1)                          #Smith, Pimblott assumptions (iso.scattering)
                        """
                        #"""
                        #Upper limit: No scatterting (Glancing collision)
                        mus           = 1                                       #cos(0 deg)
                        mu            = mus*mu - np.sqrt(1-mus**2) * np.sqrt(1-mu**2) * math.cos(gamma)
                        #"""


                        """
                        eloss_cumsum  = np.cumsum(ELOSS_int(interval_exc))             #Enrgy loss from ELF (k=0)
                        if eloss_cumsum[-1] == 0:
                            eloss_cdf = np.zeros(len(eloss_cumsum))
                        else:
                            eloss_cdf = eloss_cumsum / eloss_cumsum[-1]
                        r_exc         = random.random()                
                        eloss_exc     = np.interp(r_exc, eloss_cdf, interval_exc)
                        """
                        #"""                                        
                        Q_cumsum      = np.cumsum(DOS_int(np.linspace(11.2,energy,100)))    #Excitation energy loss from new definition
                        Q_cdf         = Q_cumsum / Q_cumsum[-1]
                        r_exc         = random.random()
                        E_bind        = np.interp(r_exc, Q_cdf, np.linspace(11.2, energy, 100))
                        eloss_exc     = random.uniform(E_bind-4.3, E_bind-0.8)
                        #"""


                        t_excitation.append(tot_t_elapsed)
                        excitation_EAD.append(exc_sp_int(energy) * Photo_eff_Tint(energy))
                        
                        energy       -= eloss_exc
                        E_exc_loss    = eloss_exc
                        E_exc_loss_data.append(E_exc_loss)
                        eloss.append(eloss_exc)
                        
                        last_condition = "excitation"
                        
                    ##############################################################################################


                else:          #energy <= 100:

                    ########################### Backward approach (Enrgy loss from SDCS) #########################
                    #"""    

                    SDCS_data = {e:np.loadtxt(f"ELF/SDCS_{e:.1f}eV_Ochkur_mod_RPA_New.csv", delimiter=",") for e in sorted(set(np.linspace(8, 100, 93).tolist() + [14.3]))}
                    #SDCS_data = {e:np.loadtxt(f"ELF/SDCS_{e:.1f}eV_NoOchkur_RPA_New.csv", delimiter=",") for e in sorted(set(np.linspace(8, 100, 93).tolist() + [14.3]))}
                    #SDCS_data = {e:np.loadtxt(f"ELF/SDCS_{e:.1f}eV_NoOchkur.csv",   delimiter=",") for e in sorted(set(np.linspace(8, 100, 93).tolist() + [14.3]))}
                    #SDCS_data = {e:np.loadtxt(f"ELF/SDCS_{e:.1f}eV_Ochkur.csv",     delimiter=",") for e in sorted(set(np.linspace(8, 100, 93).tolist() + [14.3]))}
                    #SDCS_data = {e:np.loadtxt(f"ELF/SDCS_{e:.1f}eV_Ochkur_mod.csv", delimiter=",") for e in sorted(set(np.linspace(8, 100, 93).tolist() + [14.3]))}
                    
                    closest_energy = min(SDCS_data.keys(), key=lambda x: abs(x - energy))
                    SDCS           = SDCS_data[closest_energy]
                    SDCS_grid      = SDCS[:, 0]
                    SDCS_pdf       = SDCS[:, 1]
                    SDCS_spl       = splrep(SDCS_grid, SDCS_pdf)
                    SDCS_int       = lambda energy: splev(energy, SDCS_spl)

                    interval_SDCS  = np.linspace(7, energy, 100)
                    eloss_cumsum   = np.cumsum(SDCS_int(interval_SDCS))
                    eloss_cdf      = eloss_cumsum / eloss_cumsum[-1]
                    r_ion          = random.random()
                    eloss_ionexc   = np.interp(r_ion, eloss_cdf, interval_SDCS)
                    
                    exc_ion_threshold = 11.16                                       
                    if eloss_ionexc > exc_ion_threshold:                           ##### IONIZATION #####
                        ionization_coll += 1
                        
                        if (eloss_ionexc <= 40):                                            
                            interval_Q = np.linspace(exc_ion_threshold, eloss_ionexc, 100)                 
                        else:
                            interval_Q = np.linspace(exc_ion_threshold, 40, 100)
                        Q_cumsum         = np.cumsum(DOS_int(interval_Q))
                        Q_cdf            = Q_cumsum / Q_cumsum[-1]    
                        r_DOS            = random.random()
                        Q_ion            = np.interp(r_DOS, Q_cdf, interval_Q)
                        
                        eloss_ion        = eloss_ionexc
                        energy_secondary = eloss_ion - Q_ion 
                        
                        ##### Determine mus & mu #####
                        gamma      = random.uniform(0, 2*math.pi)
                        """
# I) Shultis & Faw
                        A          = 1
                        mus        = 0.5*((A+1)*np.sqrt((energy-eloss_ion)/energy)-Q_ion*A/np.sqrt(energy*(energy-eloss_ion)))
                        mu         = mus*mu - np.sqrt(1-mus**2) * np.sqrt(1-mu**2) * math.cos(gamma)
                        """
                        
                        """
# II) Worner, 2022
                        mus        = random.uniform(np.cos(np.pi/4), np.cos(0))                      # 0-pi/4 for < 50 eV (Worner, 2022)
                        mu         = mus*mu - np.sqrt(1-mus**2) * np.sqrt(1-mu**2) * math.cos(gamma)
                        """
                        
                        #"""
# III) Conservation of momentum
                        p0         = np.sqrt(2 * mass_e * (energy               * eV2J))    #Primary (before)
                        p1         = np.sqrt(2 * mass_e * ((energy - eloss_ion) * eV2J))    #Primary (after)
                        p2         = np.sqrt(2 * mass_e * (energy_secondary     * eV2J))    #Secondary
                        theta2_rad = random.uniform(0, 2*np.pi)                             #angle of secondary (random)
                        p2x        = p2 * np.cos(theta2_rad)
                        p2y        = p2 * np.sin(theta2_rad)
                        p1x        = p0 - p2x
                        p1y        = -p2y
                        theta1_rad = np.arctan2(p1y, p1x)                                   #angle of primary                        
                        mus        = np.cos(theta1_rad)
                        mu         = mus*mu - np.sqrt(1-mus**2) * np.sqrt(1-mu**2) * math.cos(gamma)
                        #"""

                        

                        #Secondary electrons detection probability
                        E_b_surf = 1.0#0.8                                                      #electron affinity at the surface (Worner, 2022)
                        E_b_bulk = 0.2                                                      #electron affinity in the bulk (Worner, 2022)
                        e_del    = 0#E_b_surf - E_b_bulk                                      #electrons are faster by e_del at the surface
                        r_theta  = random.uniform(0, np.pi)                                 #approaching angle
                        
                        #Energy loss rate
                        ELR_eV  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                        ELR_nm  = [0.9, 1.0, 1.7, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 6.0, 8.0, 9.0, 9.2, 9.4, 9.6, 9.8, 9.9, 10.0]
                        ELR_spl = splrep(ELR_eV, ELR_nm)                                           #distance travel until become 0.2eV (e_cutoff)
                        ELR_int = lambda energy: (energy - e_cutoff) / splev(energy, ELR_spl)           #energy loss per unit nm (eV/nm)
                        energy_secondary_surface = energy_secondary - (ELR_int(energy_secondary) * pos[2])
                        
                        if E_b_surf < ((energy_secondary_surface + e_del) * (np.cos(r_theta))**2):    #Transmission probability
                            T_d = 4 *      np.sqrt(1 - (E_b_surf / ((energy_secondary_surface + e_del) * (np.cos(r_theta))**2)))        \
                                  /   (1 + np.sqrt(1 - (E_b_surf / ((energy_secondary_surface + e_del) * (np.cos(r_theta))**2))))**2
                        else:
                            T_d = 0
                        #EELS_second_MFP = 1 / total_int(energy_secondary) / 3.125e22 * 1e16 * 1e7
                        #P_second        = np.exp(-pos[2] / EELS_second_MFP)                #Probability of reaching to the vacuum-ASW interface
                        r_second = random.random()
                        
                        if energy_ini == 14.3:
                            detect_second = 0.024
                        elif energy_ini == 19:
                            detect_second = 0.0222*3                                        #Anisotropic escape --> more (x3) detection (2.22% * 3 = 6.66%)
                            
                        if r_second <= (T_d * detect_second):                               #Escaped * Detected
                            EELS_second_coll += 1
                            EELS_second_energy.append(energy_secondary_surface)

                        
                        


                        Q                = Q_ion                                            #for G Value calculation
                        cosine           = mus
                        E_ion            = energy
                        E_ion_loss       = eloss_ion * 100 / energy                         #%
                        MT               = abs(2*np.pi/(6.626E-34/(np.sqrt(2*9.11E-31*energy*1.6E-19))*18897259886)*np.sqrt(2-2*mus))
                        
                        Q_data.append(Q)
                        cosine_data.append(cosine)
                        E_ion_data.append(E_ion)
                        E_ion_loss_data.append(eloss_ion)
                        MT_data.append(MT)
                        T_d_data.append(T_d)
                        
                        energy          -= eloss_ion                                             
                        
                        t_ionization.append(tot_t_elapsed)                        
                        eloss.append(eloss_ion)
                        energy_second.append(energy_secondary)
                        
                        last_condition = "ionization"


                        """
                        #Auger electron detection probability
                        mu_auger     = 3.0
                        var_auger    = 1.0
                        sig_auger    = math.sqrt(var_auger)
                        energy_auger = np.random.normal(mu_auger, sig_auger)
                        E_b_surf = 0.8                                                      #electron affinity at the surface (Worner, 2022)
                        E_b_bulk = 0.2                                                      #electron affinity in the bulk (Worner, 2022)
                        e_del    = E_b_surf - E_b_bulk                                      #electrons are faster by e_del at the surface
                        r_theta  = random.uniform(0, np.pi)                                 #approaching angle
                        if E_b_surf < ((energy_auger+e_del) * (np.cos(r_theta))**2):        #Transmissino probability
                            T_d = 4 *      np.sqrt(1 - (E_b_surf / ((energy_auger+e_del) * (np.cos(r_theta))**2)))        \
                                  /   (1 + np.sqrt(1 - (E_b_surf / ((energy_auger+e_del) * (np.cos(r_theta))**2))))**2
                        else:
                            T_d = 0
                        if energy_auger > E_b_surf:
                            r_auger = random.random()                      
                            if (r_auger <= 0.5 * T_d * 0.005):       #50% produced (Meesungnoen and JPJG, 2005) * T_d% escaped * 0.5% detected
                                EELS_auger_coll += 1
                                EELS_second_energy.append(energy_auger)
                        """

                    
                    else:                                                          ##### EXCITATION #####
                        excitation_coll += 1
                        excitation_EAD_coll += int(exc_sp_int(energy) * Photo_eff_Tint(energy))  #Don't update depth here. Electrons are ADDED &
                                                                                                 #track NOT terminated in case of Exc-EAD.
                        #"""
#Worner's approach (fixed border line e.g., 10 eV)
                        eloss_exc      = eloss_ionexc
                        #"""
                        
                        #100 eV case testing
                        #SDCS         = SDCS_data[energy_ini]        #Glancing --> k=0
                        #SDCS_spl     = splrep(SDCS[:, 0], SDCS[:, 1])
                        #SDCS_int     = lambda energy: splev(energy, SDCS_spl)
                        #eloss_cumsum = np.cumsum(SDCS_int(np.linspace(7, energy-0.8, 100)))
                        #eloss_cdf    = eloss_cumsum / eloss_cumsum[-1]
                        #r_exc        = random.random()
                        #eloss_exc    = np.interp(r_exc, eloss_cdf, interval)

                        
                        """
#Our approach (based on ion.eff & physical constraints)
                        Q_cumsum     = np.cumsum(DOS_int(np.linspace(exc_ion_threshold, energy, 100)))          #Excitation enrgy loss from new definition
                        Q_cdf        = Q_cumsum / Q_cumsum[-1]
                        r_exc        = random.random()
                        E_bind       = np.interp(r_exc, Q_cdf, np.linspace(exc_ion_threshold, energy, 100))
                                                
                        #eloss_exc    = random.uniform(E_bind - 4, exc_ion_threshold) 
                        eloss_exc    = random.uniform(E_bind - 4, E_bind) 
                        """

                        
                        energy          -= eloss_exc
                        E_exc_loss       = eloss_exc
                        E_exc_loss_data.append(E_exc_loss)
                        eloss.append(eloss_exc)
                        
                        last_condition = "excitation"
    
                    ##############################################################################################


                """
                #Test Worner's Secondary electron energy
                r_secondary_Worner      = random.random()
                energy_secondary = cumWorner[cumWorner[:,1] > r_secondary_Worner, 0][0] + (random.random() - 0.5)
                if (energy_secondary > energy):
                    energy_secondary = energy
                secondary.append(energy_secondary)
                eloss.append(energy - energy_secondary)                        #sample from secondary electron energy distribution
                """    




        elif (r_type < (inl_val + cap_val) / tot_val):                        #TNA (= DEA + EAD)
            cap_coll  += 1                                                        #number of tracks terminated by capture
            """
            if (energy < 4):                                     #TRAPPING
                trap_coll += 1
                eloss_trap = energy                                  #Track terminated
                eloss.append(eloss_trap)
                depth.append(np.sqrt(np.sum(pos**2)))                #update 'depth' when trapped
            """                    
            #else:                                               
            r_EAD = random.random()                
            if (r_EAD <= 0.85):                                  #85% of TNA undergo EAD (Goursaud, 1976)
                TNA_EAD_coll  += 1
                eloss_TNA_EAD  = energy                            
                eloss.append(eloss_TNA_EAD)
                t_TNA_EAD.append(tot_t_elapsed)
                mu_TNA_EAD, sigma_TNA_EAD = 1.0, 0.2            #ejection length
                s_TNA_EAD = np.random.normal(mu_TNA_EAD, sigma_TNA_EAD)
                depth.append(s_TNA_EAD)                         #take sampled value (avg 0.75 nm) for 'depth' when track terminated by TNA-EAD.
            else:
                DEA_coll  += 1                
                eloss_DEA  = energy                
                eloss.append(eloss_DEA)
                                                            #Don't update depth. Electrons are GONE and track terminated in case of TNA-DEA.
            TNA_EAD.append(cap_coll*0.85)
            energy = 0

            last_condition = "capture"


        else:                                                                 #ELASTIC
            elastic_coll += 1
            eloss_el      = 0
            eloss.append(eloss_el)
            
            last_condition = "elastic"




            
            if random.random() <= 0.05:                                      #iso component : aniso component = 1 : 19
                #"""
                gamma         = random.uniform(0, 2*math.pi)
                mu            = random.uniform(-1, 1)                        #Smith, Pimblott assumptions (iso.scattering)
                #"""
            else:
                #"""
                #Worner's ADCS (E dependent) sampling concept
                interval_mu         = np.linspace(0, 180, 1000)
                elastic_ADCS_cumsum = np.cumsum(elastic_ADCS_int(interval_mu))
                elastic_ADCS_cdf    = elastic_ADCS_cumsum / elastic_ADCS_cumsum[-1]
                r_mu                = random.random()
                mu                  = np.cos(np.radians(np.interp(r_mu, elastic_ADCS_cdf, interval_mu)))
                #"""            
            
            
            
        

        energy_loss = eloss
        

        t_travel.append(tot_t_elapsed)
        dist_travel.append(tot_distance) 
        energy_evol.append(energy)
        ifcaptured.append(trap_coll + DEA_coll + TNA_EAD_coll)


        #########Use for primary track ONLY#########
        e_aq_Gval_Ion.append(ionization_coll)
        e_aq_Gval_Exc_EAD.append(excitation_EAD_coll)
        e_aq_Gval_TNA_EAD.append(TNA_EAD_coll)

                
        """
        if 'Q' in locals():                              # Check: Seems keep appending regardless of update
            Q_data.append(Q)
        if 'cosine' in locals():
            cosine_data.append(cosine)
        if 'E_ion' in locals():
            E_ion_data.append(E_ion)
        if 'E_ion_loss' in locals():
            E_ion_loss_data.append(E_ion_loss)
        if 'E_exc_loss' in locals():
            E_exc_loss_data.append(E_exc_loss)
        if 'E_inel_loss' in locals():
            E_inel_loss_data.append(E_inel_loss)
        if 'MT' in locals():
            MT_data.append(MT)
        
        E_loss_EELS_back_specific_exc_data.append(E_loss_EELS_back_specific_exc)
        E_loss_EELS_back_specific_ion_data.append(E_loss_EELS_back_specific_ion)
        """
        
        pos_evol.append(pos.copy())
        

        if (energy <= e_cutoff) and (energy != 0):
            therm_num += 1
            depth.append(np.sqrt(np.sum(pos**2)))                           #update 'depth' only when thermalized

    #######################################################END WHILE LOOP#################################################################    

        #print(therm_num)
        #meanDOSD.append(np.mean(DOSDloss))                                      #Use to comapre mean energy loss data (Pimblott 2002)
    
    if detect_min < mu < detect_max:                             #for specific angle: +-2 deg (cf. cos(121 deg) = -0.515)
        pos_evol_specific = pos_evol.copy()

    count_all += inelastic_coll+trap_coll+excitation_coll+ionization_coll+TNA_EAD_coll+elastic_coll+DEA_coll+DOSD_coll+excitation_EAD_coll
    count_cap += DEA_coll + TNA_EAD_coll + trap_coll
    
    return pos, depth, tot_distance, dist_travel, tot_t_elapsed, t_travel, energy_evol, inelastic_coll, trap_coll, excitation_coll,   \
           ionization_coll, cap_coll, DEA_coll, TNA_EAD_coll, elastic_coll, ion_init, ion_loss, count_all, count_cap, therm_num,      \
           ifcaptured, energy_incident, energy_loss, secondary, DOSD_coll, excitation_EAD_coll, meanDOSD, eloss_inelastic_per_sample, \
           Q_data, cosine_data, E_ion_data, E_ion_loss_data, E_inel_loss_data, E_exc_loss_data, e_aq_Gval_Ion, e_aq_Gval_Exc_EAD,     \
           e_aq_Gval_TNA_EAD, excitation_EAD, energy_second, t_ionization, t_excitation, t_TNA_EAD, TNA_EAD, MT_data, pos_evol,       \
           EELS_back_coll, EELS_through_coll, E_loss_EELS_back, E_loss_EELS_back_specific, surface_reflect_coll, EELS_second_coll,    \
           EELS_second_energy, pos_evol_specific, E_loss_EELS_back_specific_exc, E_loss_EELS_back_specific_ion, T_d_data