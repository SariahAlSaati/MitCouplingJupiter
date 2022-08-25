import hyperparameters as hp
import numpy as np
import os.path
from scipy.interpolate import interp1d
np.seterr(all='ignore')


"""Jupiter Transplanet Atmosphere Model (JTAM)

Input:
 1. Alt_km:  altitude in kilometers (km), [*]
 2. T0:          The thermosphere temperature at stratopause  (units: K, [*])
 3. Ta:          Temperature at za, altitude of temerature profile junction
 4. Tinf:       The thermosphere temperature at infinite altitudes (units: K, [*])
 5. Tr:          The shape parameter of the temperature profile ([*]).
 6. dTdz_a:  dT/dz at z = za  ([*]).
 7. z0:          average stratopause  height (fitting parameter, mainly for T) ([*]).
 8. za:           altitude of temerature profile junction
 9. Zh_km:   the altitude of the turbopause ([*]).
10. Zpeak_km:  The corresponding altitude where the maximum density is found
                     for the densities of H and C2H2 ([*,*]).
11. Zd_km:   The transition altitude for smoothing the piecewise
                    function for H and C2H2 ([*,*]).
12. Hd_km:   The scale height for the smoothing function for H and C2H2 ([*,*]).
13. n_lm_cm3: The density at the altitude of Zl (Zl = Za) for each species
                    (H2, H, He, CH4, C2H2), [*,*,*,*,*].
14. Npeak_cm3: The maximum density for H and C2H2 ([*,*]).
15. Fwhm_km:   Full width half maximum of the fitting Gaussian function of H and C2H2. [*, *].


The main function are:
(1) Load the basic/original JTAM model for 5 species
(2) Modify the H and C2H2 species using piecewise functions."""


def fksi(z1, z2):
    global Rp_km
    ksi = (z1 - z2)*(Rp_km + z2)/(Rp_km + z1)  # eq A4a
    dksi = ((Rp_km + z2)/(Rp_km + z1))**2
    return ksi, dksi


def SUB_JTAM_ORIGINAL(Alt_km, T0, Ta, Tinf, Tr, dTdz_a, z0, za, Zh_km,  n_lm_cm3):
    global Rp_km
    nb_spc = 5  # species number

    # physical constants
    Rp_km = 71492.
    G_o = 24.79  # gravitationnal field at Jupiter surface
    Rg = 8.314e-03  # universal gas constant

    # input parameters
    z = Alt_km[:]  # altitude in km
    Nn = np.zeros((len(z), nb_spc))  # number densities

    # A reference altitude: Zl, here we take Zl = Za for temperature and density profiles
    zl = za                        # fixed parameter, now we set to equal to za
    Tl = Ta                        # Temperature at zl
    dTl = dTdz_a                   # dT/dz | (z = zl = za)   2.176
    sig = dTl/(Tinf - Tl)
    ksia, dfksia = fksi(za, zl)

    # some coefficients for T
    ksi0 = fksi(z0, za)[0]
    Ta = Tinf - (Tinf - Tl)*np.exp(-sig*ksia)
    dTa = (Tinf - Ta)*sig*dfksia
    T12 = T0 + Tr*(Ta - T0)
    Td = 2/3*ksi0*dTa/Ta**2-3.11111*(1./Ta - 1./T0) + 7.11111*(1./T12 - 1./T0)
    Tc = ksi0*dTa/(2.*Ta**2) - (1./Ta - 1./T0) - 2.*Td
    Tb = (1./Ta - 1./T0) - Tc - Td
    v = [Td/7., Tc/5., Tb/3., 1./T0, 0]
    v[-1] = -sum(v)

    # (1) Calculate the Temperature T
    x = [1-fksi(zi, za)[0]/ksi0 for zi in z]
    vx = np.polyval(v, x)
    ksi = [fksi(zi, zl)[0] for zi in z]

    # the main function for T
    Tn = [1./(1./T0 + Tb*xi**2 + Tc*xi**4 + Td*xi**6)
          for xi in x]   # Equation A1b
    l_i = np.where(z >= za)[0]
    if len(l_i) != 0:
        for i in l_i:
            Tn[i] = Tinf - (Tinf - Tl)*np.exp(-sig*ksi[i]
                                              )          # Equation A1a

    # (2) Calculate the Densities n
    # Note here also we take zl = za: the turbopause as the reference altitude.
    M0_bar = 2.309               # average molecular mass
    gs = G_o*1.e-3

    # M: molecular mass of H2, H, He, CH4
    # A: a parameter defined like 28 / (28.95 - M)
    # H1: scale height of the correction in km (fitting parameter, A20b)
    # H2: scale height of the correction in km (fitting parameter)
    # R2: density correction parameter (fitting parameter)
    # omega: mixing ratio relative to H2 (fitting parameter)
    # z1: altitude where log C1 is R/2 (fitting parameter)
    # z2: altitude where log density correction is R2/2 (fitting parameter)
    # alfa: thermal diffusion coefficient
    # Zh_km: Zh is the altitude of the turbopause
    # =====================Note===========================
    # (1) here we take R1 = 0 and R2 = 0, i.e., we ignore the density
    # correction effects, and coefficients C1 and C2 constantly equals to 1 in the model.
    # Thus, the omega and  Nm(Zl, M_H2) terms are deleted.
    # (2) In summary, the H1, H2, omega, nm_zl_H2, R1, R2, z1, z2, C1, C2 can
    # be deleted if needed.
    # ====================================================

    M = [2, 1, 4, 16, 26]
    A = [2.3761, 2.1815, -1.1827, -0.1461, -0.1461]
    H1 = [1.5301, 1.4491, 2.8176, 1.4111, 1.4111]
    H2 = [-11.4875, -10.8558, -2.2887, -4.5763, -4.5763]
    # omega = [0.5313   ,0.5113   ,0.0025   ,0.0169   ,0.0169   ]
    # nm_zl_H2 = n_lm_cm3(1)*1.0e6     # Nm(Zl, M_H2)
    R1 = [0, 0, 0, 0, 0]    # log(omega.*nm_zl_H2./nm_lm )
    R2 = [0, 0, 0, 0, 0]  # [0.6884   ,0.7361   ,-0.3853  ,0.4146   ,0.4146   ]
    z1 = [20.5966, 19.4444, 22.3889, 20.1195, 20.1195]
    z2 = [97.5430, 98.1747, 295.9705, 111.2177, 111.2177]
    alfa = [0., 0., 0., 0., 0.]

    gl = gs/(1. + zl/Rp_km)**2
    ga = gs/(1. + za/Rp_km)**2
    gamma20 = M0_bar * gl/(sig*Rg*Tinf)
    gamma10 = M0_bar * ga*ksi0/Rg
    gamma2 = [M[i] * gl/(sig*Rg*Tinf)
              for i in range(len(M))]       # Equation A16a
    gamma1 = [M[i] * ga*ksi0/Rg for i in range(len(M))]  # Equation A16b

    # x at altitude of Zh: turbopause
    xh = 1-fksi(Zh_km, za)[0]/ksi0
    vxh = np.polyval(v, xh)
    ksih = fksi(Zh_km, zl)[0]

    Th = 1./(1./T0 + Tb*xh**2 + Tc*xh**4 + Td*xh**6)

    D_hm = [(Tl/Ta)**gamma2[i] * np.exp(-sig*gamma2[i] * ksia) *
            np.exp(gamma1[i] * vxh) for i in range(len(gamma2))]
    D_hm0 = (Tl/Ta)**gamma20 * np.exp(-sig *
                                      gamma20*ksia) * np.exp(gamma10 * vxh)

    if (Zh_km >= za):
        Th = Tinf - (Tinf - Tl)*np.exp(-sig*fksi(Zh_km, zl)[0])
        D_hm = [(Tl/Th)**gamma2[i] * np.exp(-sig*gamma2[i]*ksih)
                for i in range(len(gamma2))]
        D_hm0 = (Tl/Th)**gamma20 * np.exp(-sig*gamma20*ksih)

    # Equation A19: coefficient independent on altitudes
    nm_zm0 = (D_hm/D_hm0) * (Tl/Th)**alfa*Tl

    # calculation for each species
    nm_lm = np.zeros(nb_spc)

    for isp in range(nb_spc):
        C1 = np.exp(R1[isp]/(1. + np.exp((z-z1[isp])/H1[isp])))
        C2 = np.exp(R2[isp]/(1. + np.exp((z-z2[isp])/H2[isp])))
        D_zm = (Tl/Ta)**gamma2[isp] * np.exp(-sig *
                                             gamma2[isp] * ksia) * np.exp(gamma1[isp] * vx)
        D_zm0 = (Tl/Ta)**gamma20 * np.exp(-sig *
                                          gamma20*ksia) * np.exp(gamma10 * vx)

        l_i = np.where(z >= za)[0]
        if len(l_i) != 0:
            for i in l_i:
                D_zm[i] = (Tl/Tn[i])**gamma2[isp] * \
                    np.exp(-sig*gamma2[isp]*ksi[i])
                D_zm0[i] = (Tl/Tn[i])**gamma20 * \
                    np.exp(-sig * gamma20 * ksi[i])

        nm_zm = nm_zm0[isp] * D_zm0/Tn
        nd_zm = [D_zm[i]*(Tl/Tn[i])**(1. + alfa[isp])
                 for i in range(len(D_zm))]

        # calculate the nm_lm (unit: m-3): Nm(Zl, M), the mixing profile at Zl = Za
        nm_lm[isp] = n_lm_cm3[isp]*1.0e6 / \
            np.exp(np.log(1.0 + (nm_zm0[isp]/Tl)**A[isp])/A[isp])

        for i in range(len(Nn)):
            Nn[i][isp] = nm_lm[isp] * \
                np.exp(np.log(nd_zm[i]**A[isp] + nm_zm[i]
                       ** A[isp])/A[isp])*(C1[i]+C2[i])/2
    return Tn, Nn


def JTAM(Alt_km, T0, Ta, Tinf, dTdz_a, Tr, z0, za, Zh_km, Zpeak_km, Zd_km, Hd_km, n_lm_cm3, Npeak_cm3, Fwhm_km):
    # Step1: Load the basic/original JTAM model for 5 species
    Tn, Nn = SUB_JTAM_ORIGINAL(
        Alt_km, T0, Ta, Tinf, Tr, dTdz_a,  z0, za, Zh_km,  n_lm_cm3)
    Nn = Nn*1e-6  # from m-3 to cm-3

    dH2_TRANS = [item[0] for item in Nn]
    dH_TRANS = [item[1] for item in Nn]
    dHe_TRANS = [item[2] for item in Nn]
    dCH4_TRANS = [item[3] for item in Nn]
    dC2H2_TRANS = [item[4] for item in Nn]
    Tn_TRANS = Tn

    # Step2: Modify the H and C2H2 species using piecewise functions.
    # (1) Hydrogen
    # ***********************************
    #                 -|Transplanet,             Z > Zd
    # N(H) =    -|Smoothing function, Z ~ Zd
    #                 -|Gauss fitting,            Z < Zd
    # ***********************************
    # (1.1) Gauss Fitting: log(N) = A*log(-(z-B)**2 / C**2)
    FW = Fwhm_km[0]
    A = np.log(Npeak_cm3[0])
    B = Zpeak_km[0]
    C = np.sqrt(FW**2 / (-4*np.log((np.log(A) - np.log(2)) / np.log(A))))
    dH_fit = np.exp(A*np.exp(-(Alt_km - B)**2 / C**2))

    # (1.2) Smoothing
    Zd = Zd_km[0]
    H = Hd_km[0]  # increase H to smooth
    rec_a = np.exp(-(Alt_km-Zd)/H) / \
        (np.exp(-(Alt_km-Zd)/H) + np.exp(+(Alt_km-Zd)/H))
    rec_b = np.exp(+(Alt_km-Zd)/H) / \
        (np.exp(-(Alt_km-Zd)/H) + np.exp(+(Alt_km-Zd)/H))

    # fitting the upper part of the curve with: log(N(H)) = A*sqrt(z) + B
    if (max(Alt_km) > 900.0):
        id_900 = np.where(Alt_km >= 900)[0]
        id_900 = id_900[0]
        id_1000 = np.where(Alt_km >= 1000)[0]
        id_1000 = id_1000[0]
        A_tmp = np.log(dH_TRANS[id_900] / dH_TRANS[id_1000]) / \
            (np.sqrt(Alt_km[id_900]) - np.sqrt(Alt_km[id_1000]))
        B_tmp = np.log(dH_TRANS[id_900]) - A_tmp*np.sqrt(Alt_km[id_900])
        dH_TRANS = [np.exp(A_tmp*np.sqrt(Alt_km[i]) + B_tmp)
                    for i in range(len(Alt_km))]

    # the modified density profile
    dH_TRANS_FIT = rec_a*dH_fit + rec_b*dH_TRANS

    # (2) C2H2 TRANSPLANET Fitting
    # ***********************************
    #                       -|Transplanet,             Z > Zd
    # N(C2H2) =    -|Smoothing function, Z ~ Zd
    #                       -|Gauss fitting,            Z < Zd
    # ***********************************

    # (2.1) Gauss Fitting: log(N) = A*log(-(z-B)**2 / C**2)
    FW = Fwhm_km[1]
    A = np.log(Npeak_cm3[1])
    B = Zpeak_km[1]
    C = np.sqrt(FW**2 / (-4*np.log((np.log(A) - np.log(2)) / np.log(A))))
    dC2H2_fit = [np.exp(A*np.exp(-(Alt_km[i] - B)**2 / C**2))
                 for i in range(len(Alt_km))]

    # (2.2) Smoothing
    Zd = Zd_km[1]
    H = Hd_km[1]
    rec_a = np.exp(-(Alt_km-Zd)/H) / \
        (np.exp(-(Alt_km-Zd)/H) + np.exp(+(Alt_km-Zd)/H))
    rec_b = np.exp(+(Alt_km-Zd)/H) / \
        (np.exp(-(Alt_km-Zd)/H) + np.exp(+(Alt_km-Zd)/H))

    # the final density profile
    dC2H2_TRANS_FIT = rec_a*dC2H2_fit + rec_b*dC2H2_TRANS

    for i in range(len(Nn)):
        Nn[i][0] = dH2_TRANS[i]
        Nn[i][1] = dH_TRANS_FIT[i]
        Nn[i][2] = dHe_TRANS[i]
        Nn[i][3] = dCH4_TRANS[i]
        Nn[i][4] = dC2H2_TRANS_FIT[i]
    Tn = Tn_TRANS

    return Tn, Nn


def Get_JovianAtm():
    # Referred to Wang et al. for more details.
    # Contact: Yuxian Wang  yxwang@spaceweather.ac.cn
    # 2020/11

    # Converted into Python by Sariah Al Saati
    # contact: sariah.al-saati@polytechnique.edu
    # 2021/04

    # Free Parameters

    Alt_km = np.linspace(100, 7000, 10000)  # km

    # Free Parameters
    # [Temperature Parameters]
    T0 = 164.2
    Ta = 398.8                                    # Temperature at za
    Tinf = 1300.0  # 970.8224
    dTdz_a = 2.2                                 # dT/dz at z = za
    # average mesopause (z0) shape parameter
    Tr = -0.14

    # [Altitude Parameters]
    # average mesopause height (fitting parameter, mainly for T)
    z0 = 151.0
    # altitude of temerature profile junction (fitting parameter)
    za = 325.0
    Zh_km = 419.0                               # turbopause altitude
    Zpeak_km = [300.0, 150.0]
    Zd_km = [270, 280]
    Hd_km = [38.6, 27.6]

    # [Density Parameters]
    n_lm_cm3 = [1.0e13, 2.5e10, 1.6e11, 5.8e7, 3.4e3]   # H2, H, He, CH4, C2H2
    Npeak_cm3 = [1.9e10, 8.0e9]  # [1e10, 1e10]
    Fwhm_km = [250, 180]         # km

    Tn, Nn = JTAM(Alt_km, T0, Ta, Tinf, dTdz_a, Tr, z0, za, Zh_km,
                  Zpeak_km, Zd_km, Hd_km, n_lm_cm3, Npeak_cm3, Fwhm_km)

    dH2_TRANS = np.array([item[0] for item in Nn])
    dH_TRANS_FIT = np.array([item[1] for item in Nn])
    dHe_TRANS = np.array([item[2] for item in Nn])
    dCH4_TRANS = np.array([item[3] for item in Nn])
    Tn_TRANS = np.array(Tn)
    dC2H2_TRANS_FIT = np.array([item[4] for item in Nn])

    # ## Save data
    filename = 'atmos'
    np.savez(hp.pathtoatmo  + filename, Alt_m=Alt_km*1e3,
             T_K=Tn_TRANS, dH2_m3=dH2_TRANS*1e6, dCH4_cm3=dCH4_TRANS)
    return Alt_km*1e3, Tn_TRANS, dH2_TRANS*1e6, dCH4_TRANS
    # units in m, K, m-3, cm-3


def Get_Atmosphere(Z_m):
    # +
    # =================================================
    # Description: Get the atmosphere from Jupiter Transplanet Atmosphere Model (JTAM)
    # Input:
    #    Z_in_meter
    # Output:
    #    N_H2, R_Z, T_H2, N_CH4
    # Example:
    #    z = linspace(2000, 8000, 1.0e4)*1.0e3     # z altitude in m
    #    nz = N_elements(z)
    #    Data0 = Get_Atmosphere_H2(Z)
    #    N_0 = transpose(Data0[0,*]) / 1.0e6  #cm-3
    #    R_0 = transpose(Data0[1,*])  # kg/m2
    # History:
    #    Written by Yuxian Wang 2020-07-25 18:32:46
    # =================================================
    # -

    # The JTAM-V6 atmosphere model results
    n = len(Z_m)
    N_H2 = np.zeros(n)          # output density
    N_Zkm = np.zeros(n)         # tmp useful  density
    R_Z = np.zeros(n)           # output R
    T_H2 = np.zeros(n)          # output T
    N_CH4 = np.zeros(n)         # output CH4

    filename = "atmos.npz"
    if not os.path.isfile(hp.pathtoatmo + filename):
        Alt_m, T_K, dH2_m3, dCH4_cm3 = Get_JovianAtm()
    else:
        atmosdata = np.load(hp.pathtoatmo + filename)
        Alt_m = atmosdata['Alt_m']
        T_K = atmosdata['T_K']
        dH2_m3 = atmosdata['dH2_m3']
        dCH4_cm3 = atmosdata['dCH4_cm3']

    # (2) Calculate R(z)
    # [>7000 km: upper brand - Alt = klog(N) + B]
    k = (Alt_m[-2] - Alt_m[-10]) / \
        (np.log10(dH2_m3[-2]) - np.log10(dH2_m3[-10]))
    b = Alt_m[-2] - k*np.log10(dH2_m3[-2])

    # [interpolate data to uniform spaced Altitudes: (1km, 10000km)]
    Uni_Z_m = np.linspace(1.0, 10000, int(5.0e4))*1.0e3   # m
    Uni_H2_m3 = np.zeros(len(Uni_Z_m))     # density
    Uni_T_K = np.zeros(len(Uni_Z_m))      # T

    f_H2_m3 = interp1d(Alt_m, dH2_m3, fill_value="extrapolate")
    Uni_H2_m3 = f_H2_m3(Uni_Z_m)
    f_T_k = interp1d(Alt_m, T_K, fill_value="extrapolate")
    Uni_T_K = f_T_k(Uni_Z_m)

    # [>3800 km]
    ind = np.where(Uni_Z_m >= 7000.0e3)[0]
    count = len(ind)
    if (count >= 1):
        Uni_H2_m3[ind] = 10**((Uni_Z_m[ind] - b) / k)
        Uni_T_K[ind] = T_K[-1]

    # (3) the output density
    f_N_H2 = interp1d(Uni_Z_m, Uni_H2_m3, fill_value="extrapolate")
    N_H2 = f_N_H2(Z_m)
    f_T_H2 = interp1d(Uni_Z_m, Uni_T_K, fill_value="extrapolate")
    T_H2 = f_T_H2(Z_m)
    f_N_CH4 = interp1d(Alt_m, dCH4_cm3, fill_value="extrapolate")
    N_CH4 = f_N_CH4(Z_m)*1.0e6   # /m3

    # (4) R_Z： the total mass density above the Z.
    for i in range(len(Z_m)):
        ind = np.where(Uni_Z_m >= Z_m[i])
        if len(ind) > 0:
            R_Z[i] = np.sum(Uni_H2_m3[ind])*(Uni_Z_m[1] - Uni_Z_m[0])

    R_Z *= 2.0*1.67e-27   # kg/m2

    return N_H2, R_Z, T_H2, N_CH4


if __name__ == '__main__':
    Get_JovianAtm()
