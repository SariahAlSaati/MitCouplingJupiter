import hyperparameters as hp
import UTLutils as ut
import UTLprepro as utprepro
import UTLcrossings as utxings
import UTLmag as utmag
import MODatmosphere as modatm
import numpy as np
from scipy.interpolate import interp1d


"""Description:
    This pro (non-public) is the ionospheric model, which is based on Hiraki and Tao. (2008)
    and Gerard et al. (2020, JGR) models, mainly to calculate the conductivities and ion/ele density
    due to the electron precipitations. More details about the model can be refered to the
    supplement of the Wang et al. paper.

    The program includes 3 main subroutines:
    1. Sub_Beam_Conductance:  beam/Maxwellian
    2. Sub_Cond_AlongJuno:  getting the data along the Juno footprints
    3. Jupiter_Ionos_Model:  Main program, one can run directly.
Input:
    The JADE/JEDI combined energy flux data are from ...
       FLUX_LOSSCONE_PJ*_South.sav
    Atmospheric output data.

Output:
    Sub_Beam_Conductance: plot the Figure of the energy-dependent Pedersen/Hall conductances
       for constant precipitating energy flux of 100 mW/m2.

    Sub_Cond_AlongJuno: Calculate the conductances and et. along the Juno footprints, and save as
       e.g., Cond_PJ03_South.sav"""


def subfun(x, qion, k, alpha, alpha1):
    # Solve the equations with Newton's method
    y = -0.5*x + 0.5*np.sqrt(x**2 + 4.0*x*alpha1*k)
    z = x + y
    gy = alpha*x*z + alpha1*x*k - qion
    ans = [gy, x, y, z]
    return ans


def Get_IonEleDensity(N_CH4, qion_in, Tn):
    # Calculate the ionospheric electrons/ions under the photochemistry balance.
    # Input: One point each time.
    #    N_CH4: number density of CH4 in m-3
    #    qion_in: production rate in /s/m3
    #    Tn: temperature in K, here assume Te = Tn
    # Output:
    #    ans = [gy, x, y, z], gy ->0 is best, x=[H3+], y=[CH5+], z=[e-]

    if (N_CH4*qion_in*Tn) < 0:
        raise ValueError('Error: Negative value detected!')

    # parameters
    qion = qion_in   # 2.0e9   # /s/m3  ionization rate
    k = N_CH4    # 1.0e10*1.0e6   # /m3  CH4 number density
    Te = Tn  # 600   # K
    # m3/s  combination of H3+ and e-
    alpha = 1.2e-7*(300.0 / Te)**0.65*1.0e-6
    alpha1 = 2.4e-9*1.0e-6   # m3/s  H3+ and CH4
    # m3/s  combination of CH5+ and e-
    alpha2 = 2.7e-7*(300.0 / Te)**0.52*1.0e-6

    alpha00 = 1.0e-13  # combination rate constant
    alpha /= alpha00
    alpha1 /= alpha00
    alpha2 /= alpha00
    N00 = 1.0e12    # number density constant
    qion /= N00**2*alpha00
    k /= N00

    x0 = 1.0e9/N00   # m3    CH4+
    dx = 1.0e7/N00
    resi = 1.0
    g0 = 1
    while (resi >= 1.0e-8) or (abs(g0) >= 1.0e-7):
        x = x0
        g1 = subfun(x+dx, qion, k, alpha, alpha1)
        g1 = g1[0]
        g0 = subfun(x, qion, k, alpha, alpha1)
        g0 = g0[0]
        gp = (g1 - g0) / dx
        x0 = x - g0 / gp
        resi = abs(x0 - x)

    ans = subfun(x0, qion, k, alpha, alpha1)
    ans = [ans[i]*N00 for i in range(len(ans))]
    return ans


def k_in_lamda(e):
    # ******************************
    # K: energy-dependent part of Lamda = Lamda0*K
    # e input as KeV
    # e in keV
    cc = [0.13+0.89*(1 - 1.1*np.tanh(np.log10(e_i)-1)) for e_i in e]
    return cc


def lamda0(xx):
    # ******************************
    # Lamda0: height-dependent part of Lamda = Lamda0*K
    xx = np.array(xx)
    lam0 = np.zeros(xx.shape)
    ind = np.where((xx > 0) & (xx <= 0.3))[0]
    if len(ind) > 0:
        for i in ind:
            lam0[i] = -669.54*xx[i]**4+536.18*xx[i]**3 - \
                159.86*xx[i]**2+18.586*xx[i]+0.5064

    ind = np.where((xx > 0.3) & (xx <= 0.825))[0]
    if len(ind) > 0:
        for i in ind:
            lam0[i] = 0.7677*xx[i]**4-5.9034*xx[i]**3 + \
                12.1190*xx[i]**2-9.7343*xx[i]+2.7471

    ind = np.where((xx > 0.825) & (xx <= 1))[0]
    if len(ind) > 0:
        for i in ind:
            lam0[i] = -0.8091*xx[i]**3+2.4516*xx[i]**2-2.4778*xx[i]+0.8354

    return lam0


def Get_IonizationRate(Alt_m, E0_keV, theta_rad_one, N_H2_m3, R_Z):
    # +
    # =================================================
    # Description:
    #    This pro is to calculate the ionization rate (/m) at the altitude of z:
    #      electrons with energy e0 impact the atmosphere with an angle of theta
    # Input:
    # Output:
    #    ionization rate: qion[E, Z]
    # Example:
    #      qion = Get_Ionization_Rate(Alt_m, E0_keV, theta_rad_one, N_H2_m3, R_Z)
    # History:
    #   Written by Yuxian Wang 2020-07-27 19:22:51
    # =================================================
    # -
    # Note that theta_rad should be only 1 value.

    # (1) Atmosphere model
    nz = len(Alt_m)
    z = Alt_m
    nee = len(E0_keV)

    N_H2 = np.array(N_H2_m3)  # m-3
    R_Z = np.array(R_Z)                # kg/m2

    # (2) Aurora precipitation particle
    e0 = E0_keV   # keV
    #qion = fltarr(N_elements(e0), nz)
    qion = np.zeros((nz, len(e0)))
    etta = np.zeros((nz, len(e0)))

    # (3) Parameterized lmd & Qion:  qion in m-1
    Eion = 0.03     # keV, newly defined Eion = 30 eV
    c0 = 6.0
    theta = theta_rad_one
    c = - np.log10(-0.0436*theta**2 + 0.0499*theta + 0.0302)
    a = (0.5/np.cos(theta) + 0.5 - 1) / ((c0+c)**2/4.0 - c0*c)
    b = - np.log10(0.75*np.cos(theta) + 0.25) / (c - 0.523)

    Rs = [3.39e-5*e0_i**1.39 for e0_i in e0]   # R0s  [30000]   [E]
    tmp = 1.0/Eion*N_H2*2.*1.67e-27   # [Z]
    k_in_lamda00 = k_in_lamda(e0)   # [E]

    for i in range(nz):
        xx = [R_Z[i]/Rs[j] for j in range(len(Rs))]
        lam0 = lamda0(xx)
        for j in range(len(e0)):
            qion[i][j] = e0[j]*tmp[i]/Rs[j] * k_in_lamda00[j]*lam0[j]

    return qion


def Sub_Beam_Conductance():
    # *********************************
    #   The routine to calculate the conductances and densities for the electron beam precipitation
    #   into the neutral atmosphere.
    #
    #   Note that the results are very sensitive to the neutral atmospheric model and the B-field.
    #   2 Steps for this simulation.
    # *********************************

    # Step 1/2: mono-beam or Maxwellian beam?
    # -------------------------------------------------
    flagb = 0  # 0 for mono-energetic beam, 1 for Maxwellian beam
    # Step 2/2: go to the main program to change to beam/Maxwellian
    # change steps 1-4: e.g., flag_eflux = 0 of the main program@Jupiter_Ionos_Model
    #
    # S1: funcflag = 1
    # S2: Eleflag = 0
    # S3: flag_eflux = 2
    # S4: yes
    # -------------------------------------------------
    # (1) Basic settings
    # e0 = [0.15, 0.5, 5, 10, 50, 100, 200, 1000]  # keV
    e0 = np.linspace(-1, 4.3, 100)
    e0 = [10**e0[i] for i in range(len(e0))]  # keV

    ParamAll = np.zeros((7, len(e0)))

    for i in range(0, len(e0)):
        e0_keV = e0[i]
        totalflux, Qs, Params = Jupiter_Ionos_Model(e0_keV=e0_keV)
        #  Params = [SIGP, SIGH, N_ele_col, SIGP3, SIGH3, N_ele_col3, N_H3p_col3]
        #  0,1,2 is without CH4, and 3, 4, 5, 6 is with CH4.
        ParamAll[:, i] = Params
        print('i = ', i, '/', len(e0))

    # dashed: without CH4, dashed: with CH4
    SPs1 = ParamAll[3, :], SHs1 = ParamAll[4, :]
    SPs3 = ParamAll[0, :], SHs3 = ParamAll[1, :]

    return SPs1, SHs1, SPs3, SHs3


def Sub_Cond_AlongJuno(PJnum, NS, NC):

    # Step 1/2: Set the time
    # ----------------------------------------
    # 0 for north, 1 for south

    # Step 2/2: go to main program check: Steps 1-4
    # ----------------------------------------

    # S1: funcflag = 2
    # S2: Eleflag = 2
    # S3: flag_eflux = 0
    # S4: deleted
    # 4min ahead and later than standard one.
    tlim = utxings.Get_MainOvalCrossUTs(
        PJnum, NS, NC, OffSet_min=hp.Offset_min)

    # Set the sample points number
    julim = np.zeros(2)
    julim[0] = ut.yxutc2jul(tlim[0])
    julim[1] = ut.yxutc2jul(tlim[1])
    Nt = int(round((julim[1] - julim[0])*86400.0 / (1.0*hp.resolution))) + 1
    # see resolution in hyperparameters.py
    juls = np.linspace(julim[0], julim[1], Nt)

    SPs = np.zeros(Nt)
    SHs = np.zeros(Nt)
    N_H3p = np.zeros(Nt)
    TEFlux = np.zeros(Nt)
    Qss = np.zeros(Nt)

    ParamAll = np.zeros((7, Nt))

    for i in range(0, Nt):
        print('i = ', i, '/', Nt-1)
        UT0 = ut.yxjul2utc(juls[i])
        Qs, totalflux, Params = Jupiter_Ionos_Model(
            PJnum=PJnum, NS=NS, NC=NC, UT0=UT0, e0_keV=3)
        # [Params = [SIGP, SIGH, N_ele_col, SIGP3, SIGH3, N_ele_col3, N_H3p_col3]
        # [0,1,2 is without CH4, and 3, 4, 5, 6 is with CH4.]

        # print("Params", Params)
        # print("totalflux", totalflux)
        # print("Qs", Qs)

        ParamAll[:, i] = Params
        TEFlux[i] = totalflux
        Qss[i] = Qs

    Qs = Qss

    filename = 'Cond_' + ut.get_name(PJnum, NS, NC)

    np.savez(hp.pathtoiono + filename, ParamAll=ParamAll,
             TEFlux=TEFlux, juls=juls, Qs=Qs)
    # This is the conductances calculated based on Taos model.
    # units are: mho, X1.0e12 /cm2, W/m2, julday.
    # ParamAll = [SIGP, SIGH, N_ele_col, SIGP3, SIGH3, N_ele_col3, N_H3p_col3]
    # SIGP is for without CH4, and SIGP3 is for with CH4 included
    return ParamAll, TEFlux, juls, Qs


def Jupiter_Ionos_Model(PJnum, NS, NC, UT0, e0_keV):
    # *********************************
    # The main program for the ionospheric model.
    # The whole program run for only one point (position or time point) each time. Cycle for
    #  time series.
    #
    #  [energies in keV & Altitudes in m]
    #  Input:
    #     PJnum, NS = 0 for North, 1 for South, currently south only.
    #     UT0: time in UT
    #     e0_keV: a characteristic energy, only for beams simulation.
    #
    #  Output:
    #     totalflux: total energy flux from JADE+JEDI in W/m2
    #     Qs: 0-1 quality index of the JADE+JEDI dataset, 1 is best
    #     Params: parameters
    #       [Params = [0-SIGP, 1-SIGH, 2-N_ele_col,
    #           3-SIGP_CH4, 4-SIGH_CH4, 5-N_ele_col_CH4, 6-N_H3p_col_CH4]
    #       [0,1,2 is without CH4, and 3,4,5,6 is with CH4.]
    # *********************************

    # (1) Basic Settings:
    #   4 STEPS IN TOTAL
    #
    # Step 1/4: Setting PJnum + NS + UT0
    # -----------------------------
    # Set the Juno location, i.e., the B-field, e-flux and Te/Tn (optional)
    # Sub_Cond_AlongJuno: do not set PJnum and UT0
    # Sub_Beam_Conductance, or directly run: set PJnum and UT0

    funcflag = 2
    # 0 for Jupiter_Ionos_Model, directly run
    # 1 for Sub_Beam_Conductance
    # 2 for Sub_Cond_AlongJuno
    if funcflag == 0:
        # Routine: Jupiter_Ionos_Model, directly run
        PJnum = 1
        NS = 0
        UT0 = '2016-08-27T12:10:00'

    if funcflag == 1:
        # Routine: Sub_Beam_Conductance
        PJnum = 1
        NS = 0
        UT0 = '2016-08-27T12:10:00'

    if funcflag == 2:
        # Routine: Sub_Cond_AlongJuno
        PJnum = PJnum
        NS = NS
        UT0 = UT0

    # Step 2/4: Setting the energy
    # -----------------------------
    Eleflag = 2
    # 0: Monoenergetic beams (E = e0), Eflux_flag = 1, 2
    # 1: Maxwellian beams (Echarac = e0), Eflux_flag = 3
    # 2: Electron Spectrum, Eflux_flag = 0
    if Eleflag == 0:
        # Monoenergetic beams
        e0 = e0_keV

    if Eleflag == 1:
        # Maxwellian beams
        Ec_keV = e0_keV    # characteristic energy
        e0 = np.linspace(0.1, 3000, int(1e3))  # keV

    if Eleflag == 2:
        # Electron Spectrum
        e0 = np.linspace(0.1, 3000, int(1e3))  # keV

    # Step 3/4: The precipitation energy flux
    # -----------------------------
    # Flag: 0 - realtime Juno JADE+JEDI EFlux, keyword: PJnum=PJnum, NS=NS, UT0=UT0, Qs=Qs_out
    # Flag: 1 - constant particle flux for mono-beams, delta function of energy: /m2/s.
    # Flag: 2 - constant energy flux for mono-beams, e.g., 100 mW/m2
    # Flag: 3 - Maxwellian distribution
    # 0,3 return: /cm2/s/eV (Prod=qion*F*dE)    1-2 return: /m2/s (Prod=qion*F)
    flag_eflux = 0
    PFlux, Qs, totalflux = Get_PrecipitationFlux(
        e0, flag_eflux, PJnum=PJnum, NS=NS, NC=NC, UT0=UT0)

    # [define altitudes]
    Alt_m = np.linspace(1, 3500, int(1e3))*1.0e3  # m
    nz = len(Alt_m)
    ne0 = len(e0)

    # (2) Get Magnetic field: |B|
    jno_xyz = ut.Get_JunoState(np.array([UT0]), 'iau')
    jno_rtp = ut.Car2SphP(jno_xyz[:, 0], jno_xyz[:, 1], jno_xyz[:, 2])
    lat0 = 90.0 - jno_rtp[:, 1]
    lon0 = jno_rtp[:, 2]       # degrees   IAU-Jupiter

    # [Position]
    PosRTP = np.zeros((3, nz))   # Positions to get the B-jrm09
    PosRTP[0, :] = 1. + Alt_m/71492.0/1000.0    # r in Rj
    PosRTP[1, :] = (90.0 - lat0) * np.ones(nz)  # theta in deg
    PosRTP[2, :] = lon0*np.ones(nz)  # theta in deg

    # [JRM09 + CAN]
    Br_jrm, Bt_jrm, Bp_jrm, Bmag = utmag.GetJupiterMag(
        PosRTP[0, :], PosRTP[1, :], PosRTP[2, :], CAN=1)
    inci_theta = np.pi/2 - np.arctan2(Br_jrm[0], Bt_jrm[0])
    # angle of the magnetic field to the radial direction.

    theta_rad_one = 0    # inci_theta, effects are little so here set to 0
    B = Bmag*1.0e-9     # T

    # Step 4/4: The B-field
    # -----------------------------
    # B = B*0.0 + 10e-4   # T  i.e., 10.0G # 1T = 10000G   (for Sub_Beam_Conductance only)

    # (3) Atmosphere H2 and CH4
    tmp = modatm.Get_Atmosphere(np.array(Alt_m))
    N_H2 = tmp[0]
    R_z = tmp[1]       # kg/m2
    T_z = tmp[2]        # K
    N_CH4 = tmp[3]  # /m3

    # (4) Production rate: qion*F*dE
    IonRate = np.zeros(nz)
    qion = Get_IonizationRate(Alt_m, e0, theta_rad_one, N_H2, R_z)

    for i in range(0, nz):
        if (flag_eflux == 1) or (flag_eflux == 2):
            # for beams   1, 2
            IonRate[i] = np.sum(PFlux[:] * qion[i, :])  # /s/m3
        else:
            # for spectrum  0, 3
            IonRate[i] = np.sum(PFlux[:] * qion[i, :]) * \
                1.0e4*(e0[1] - e0[0])*1.0e3  # /s/m3

    # (5) Calculate the ion/e density (CH5+)
    # #Method 1: Gerard+2020 without CH5+
    # assume T(e) = T(H2)
    alpha = 1.2e-7*(300.0/T_z)**0.65*1.0e-6      # Sundstrom+1994
    N_ele = np.sqrt(IonRate / alpha)                       # Gerard+2020
    N_H3p = N_ele
    N_CH5p = T_z*0.0   # No CH5+

    # #Method 2: Include CH5+ ions
    # [e-] = [CH5+] + [H3+]
    print('# CH5+ iteration: Begin')

    Nds = np.zeros((4, nz))
    for i in range(0, nz):
        N_CH4_tmp = N_CH4[i]
        IonRate_tmp = IonRate[i]
        T_z_tmp = T_z[i]

        # All the combination coefficients are inside the pro below.
        # There is an iteration procedure inside the pro.
        ans = Get_IonEleDensity(N_CH4_tmp, IonRate_tmp, T_z_tmp)
        Nds[:, i] = ans
    N_H3p3 = Nds[1, :]
    N_CH5p3 = Nds[2, :]
    N_ele3 = Nds[3, :]

    print('# CH5+ iteration: END')

    # (5) Get the Sig_P and Sig_H
    # electron + ion (H3+) + neutral (H2)

    # [Constants]
    q = 1.6e-19  # C
    me = 9.1e-31
    mp = 1.67e-27      # kg
    kB = 1.38e-23    # Boltzmann Constant
    k2ev = 11600.
    a0 = 5.2917e-11   # Bohr radius in m

    # [Collision frequency]
    wce = B*q/me
    wci = B*q/(mp*3)
    wcih = B*q/(mp*17.0)  # CH5+
    mion = 3.0*mp   # H3+
    mh2 = 2.0*mp    # H2

    # From Gerard+ private communication
    vin = 2.6e-9*N_H2*np.sqrt(0.7 / 1.2)/1.0e6   # /s
    ven = N_H2/1.0e6*2.5e-9*(1.0 - 1.35e-4*T_z)*T_z**0.5

    # [Sig_PH in mho/m]
    Sig_P = N_ele*q/B * wce*ven/(ven**2 + wce**2) + \
        N_H3p*q/B * wci*vin/(vin**2 + wci**2)

    Sig_P3 = N_ele3*q/B * wce*ven/(ven**2 + wce**2) + N_H3p3*q/B * wci*vin/(
        vin**2 + wci**2) + N_CH5p3*q/B * wcih*vin/(vin**2 + wcih**2)

    # Sig_P_e = N_ele3*q/B *wce*ven/(ven**2 + wce**2)
    # Sig_P_H3p = N_H3p3*q/B * wci*vin/(vin**2 + wci**2)
    # Sig_P_CH4p = N_CH5p3*q/B * wcih*vin/(vin**2 + wcih**2)

    Sig_H = N_ele*q/B * wce**2/(ven**2 + wce**2) - \
        N_H3p*q/B * wci**2/(vin**2 + wci**2)
    Sig_H3 = N_ele3*q/B * wce**2/(ven**2 + wce**2) - N_H3p3*q/B * wci**2/(
        vin**2 + wci**2) - N_CH5p3*q/B * wcih**2/(vin**2 + wcih**2)

    # [height integrated conductance]
    SIGP = np.sum(Sig_P)*(Alt_m[1] - Alt_m[0])   # in mho
    SIGP3 = np.sum(Sig_P3)*(Alt_m[1] - Alt_m[0])   # in mho
    SIGH = np.sum(Sig_H)*(Alt_m[1] - Alt_m[0])   # in mho
    SIGH3 = np.sum(Sig_H3)*(Alt_m[1] - Alt_m[0])   # in mho

    # [height integrated density (column density cm-2)]
    N_ele_col = np.sum(N_ele)*(Alt_m[1] - Alt_m[0]) / \
        1.0e4/1.0e12  # X1.0e12 /cm2
    N_ele_col3 = np.sum(N_ele3) * \
        (Alt_m[1] - Alt_m[0])/1.0e4/1.0e12  # X1.0e12 /cm2
    N_H3p_col3 = np.sum(N_H3p3) * \
        (Alt_m[1] - Alt_m[0])/1.0e4/1.0e12  # X1.0e12 /cm2

    Params = [SIGP, SIGH, N_ele_col, SIGP3, SIGH3, N_ele_col3, N_H3p_col3]

    return totalflux, Qs, Params


def Get_PrecipitationFlux(E_keV_in, flag, Ec_keV=None, PJnum=None, NS=None, NC=None, UT0=None):
    # +
    # =================================================
    # Description:
    #    This pro is to get the auroral precipitation particle flux.
    #    Pflux: particle number for unit time, unit area, and unit energy around E0： 1/cm2/s/eV
    #
    #    Flag: 0 - realtime Juno JADE+JEDI EFlux, keyword: PJnum=PJnum, NS=NS, UT0=UT0
    #    Flag: 1 - constant energy flux for mono-beams, delta function of energy: /m2/s.
    #    Flag: 2 - constant energy flux for mono-beams, e.g., 100mW/m2
    #    Flag: 3 - Maxwellian distribution
    #
    # Input:
    #    E_keV_in: The energy of particles in unit of KeV.
    #    flag: 0, 1, 2
    #    Ec_keV: characteristic energy
    #    PJnum: PJ number
    #    NS: 0 for north, 1 for south
    #    NC: numero of the crossing
    #    UT0: UTC time
    #
    # Output:
    #    Qs: quality index (0 - 1 = worst - best)
    #    totalflux: total energy flux in W/m2
    #
    # Return:
    #    Particle flux: 1/eV/cm2/s defined as omni differential particle flux
    #    (no sr information.).
    #    0 return: /cm2/s/eV (Prod=qion*F*dE)    1-2 return: /m2/s (Prod=qion*F)
    #
    # Example:
    # History:
    #   Written by Yuxian Wang 2020-09-02 17:26:25
    # =================================================
    # -

    E = np.array(E_keV_in).astype(float)

    if (flag == 0):
        if NS != 0 and NS != 1:
            raise ValueError('Please define value for NS. (flag = 0)')
        if not PJnum:
            raise ValueError('Please define value for PJnum. (flag = 0)')
        if not UT0:
            raise ValueError('Please define value for UT0. (flag = 0)')
        if NC != 0 and NC != 1:
            raise ValueError('Please define value for NC. (flag = 0)')
        Qs = 0
        # (1) Load realtime JADE+JEDI Eflux
        dtarr, E0_KEV, PFLUX, PFLUX_JADE_BIN, PFLUX_JEDI_BIN = utprepro.eflux(
            PJnum, NS, NC)
        JULS_BIN = ut.datetime2julian(dtarr)

        # [time clipping: Sample one time]
        jul0 = ut.yxutc2jul(UT0)
        ind = np.where(JULS_BIN >= jul0)[0]
        count1 = len(ind)

        PFLUX_full = np.zeros(E.shape)   # PFLUX_full(e0)
        if count1 >= 1:
            ind = ind[0]
            PFLUX0 = PFLUX[:, ind]

            # [Interpolate the valid periods: Emin<E<Emax]
            ind2 = np.where(np.isfinite(PFLUX0))[0]
            count2 = len(ind2)
            if count2 == 0:
                raise ValueError(
                    'Error: All the energy flux at this time is NAN.')
            Emin = E0_KEV[ind2[0]]
            Emax = E0_KEV[ind2[-1]]
            Fmax = PFLUX0[ind2[-1]]

            ind3 = np.where(E <= Emax)[0]
            f_flux = interp1d(E0_KEV[ind2], PFLUX0[ind2],
                              fill_value="extrapolate")
            PFLUX_full[ind3] = f_flux(E[ind3])

            # [Append with kappa distribution: E>Emax]
            ind4 = np.where(E > Emax)[0]
            count3 = len(ind4)
            if count3 >= 1:
                x0 = np.log10(Emax)
                y0 = np.log10(Fmax)
                k = -5.1
                B = y0 - k*x0
                PFLUX_full[ind4] = 10**B*E[ind4]**k

            # [Calculate quality index：Qs = JADE(valid points / total)*JEDI(valid points / total)]
            test_jade = PFLUX_JADE_BIN[:, ind]
            ind_jade = np.where(np.isfinite(test_jade))[0]
            count11 = len(ind_jade)
            test_jedi = PFLUX_JEDI_BIN[:, ind]
            ind_jedi = np.where(np.isfinite(test_jedi))[0]
            count22 = len(ind_jedi)
            Qs = count11/64.0 * count22/21.0

        # Make sure no negative values from interpolation
        indf = np.where(PFLUX_full <= 0)[0]
        if len(indf > 0):
            PFLUX_full[indf] = 1.0e-9
        totalflux = np.sum(PFLUX_full*E*(E[1]-E[0])*1.6e-9)   # W/m2
        Pflux = PFLUX_full   # /cm2/s/eV

    if (flag == 1):
        # (2) load constant Particle Flux.
        Pflux = 6.25e12   # /m2/s, energy flux, a delta function of Energy
        totalflux = np.sum(Pflux*E*1000.0*1.6e-19)    # W/m2  delta function

    if (flag == 2):
        # (3) load constant EFlux.
        # fixed as 100 mW/m2 for all mono-energetic beams
        # /m2/s, energy flux, a delta function of Energy
        Pflux = 100.0e-3/1.6e-19/(E[0]*1000.0)
        totalflux = 100.0e-3  # W/m2

    if (flag == 3):
        # (4) Maxwellian distribution
        if not Ec_keV:
            Ec_keV = 100  # characteristic energy
        phi0 = 3.125e5          # cm-2/s/eV    initial value, will iterate to reach 100 mW/m2
        totalflux_final = 0.1  # W/m2
        alpha = Ec_keV  # 100 keV
        Pflux = phi0 * E/alpha*np.exp(-E / alpha)   # /s/cm2/eV

        # iterate to 100 mW/m2
        Eflux = np.sum(Pflux*1.0e4*(E[1]-E[0])*1.0e3*E*1000.0*1.6e-19)  # W/m2
        phi0 = phi0*totalflux_final/Eflux
        Pflux = phi0 * E/alpha*np.exp(-E / alpha)
        totalflux = np.sum(Pflux*E*(E[1]-E[0])*1.6e-9)   # W/m2

    try:
        Qs
    except NameError:
        return Pflux, totalflux
    else:
        return Pflux, totalflux, Qs


def totalFlux(PJnum, NS, NC):
    tlim = utxings.Get_MainOvalCrossUTs(PJnum, NS, NC)
    julim = np.zeros(2)
    julim[0] = ut.yxutc2jul(tlim[0])
    julim[1] = ut.yxutc2jul(tlim[1])
    Nt = int(round((julim[1] - julim[0])*86400.0 / (1.0*hp.resolution))) + 1
    juls = np.linspace(julim[0], julim[1], Nt)
    TEFlux = np.zeros(Nt)
    TFlux = np.zeros(Nt)
    e0 = np.linspace(0.1, 3000, int(1e3))  # in keV
    dtarr, E0_KEV, PFLUX, PFLUX_JADE_BIN, PFLUX_JEDI_BIN = utprepro.eflux(
        PJnum, NS, NC)
    JULS_BIN = ut.datetime2julian(dtarr)
    PFLUX_full = np.zeros(e0.shape)
    for i in range(0, Nt):
        UT0 = ut.yxjul2utc(juls[i])
        jul0 = ut.yxutc2jul(UT0)
        ind = np.where(JULS_BIN >= jul0)[0]
        count1 = len(ind)
        if count1 >= 1:
            ind = ind[0]
            PFLUX0 = PFLUX[:, ind]
            ind2 = np.where(np.isfinite(PFLUX0))[0]
            count2 = len(ind2)
            if count2 == 0:
                raise ValueError(
                    'Error: All the energy flux at this time is NAN.')
            Emin = E0_KEV[ind2[0]]
            Emax = E0_KEV[ind2[-1]]
            Fmax = PFLUX0[ind2[-1]]

            ind3 = np.where(e0 <= Emax)[0]
            f_flux = interp1d(E0_KEV[ind2], PFLUX0[ind2],
                              fill_value="extrapolate")
            PFLUX_full[ind3] = f_flux(e0[ind3])

            # [Append with kappa distribution: E>Emax]
            ind4 = np.where(e0 > Emax)[0]
            count3 = len(ind4)
            if count3 >= 1:
                x0 = np.log10(Emax)
                y0 = np.log10(Fmax)
                k = -5.1
                B = y0 - k*x0
                PFLUX_full[ind4] = 10**B*e0[ind4]**k
        # Make sure no negative values from interpolation
        indf = np.where(PFLUX_full <= 0)[0]
        if len(indf > 0):
            PFLUX_full[indf] = 1.0e-9
        totalEflux = np.sum(PFLUX_full*e0*(e0[1]-e0[0])*1.6e-9)   # W/m2
        totalflux = np.sum(
            PFLUX_full*(e0[1]-e0[0])*1.6e-19*1e6*1e3*1e4)  # microA/m2
        TEFlux[i] = totalEflux
        TFlux[i] = totalflux
    nante = np.where(np.isnan(TEFlux))[0]
    notnante = np.where(~np.isnan(TEFlux))[0]
    for i in nante:
        id = notnante[notnante < i]
        lend = len(id)
        if lend == 0:
            id = 0
        else:
            id = id[-1]
        iu = notnante[notnante > i]
        lenu = len(iu)
        if lenu == 0:
            iu = lenu
        else:
            iu = iu[0] + 1
        TEFlux[i] = np.nanmean(TEFlux[id: iu])
        TFlux[i] = np.nanmean(TFlux[id: iu])
    return ut.julian2datetime(juls), TEFlux, TFlux


if __name__ == '__main__':
    pass