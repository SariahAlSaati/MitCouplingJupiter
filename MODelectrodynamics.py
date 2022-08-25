import hyperparameters as hp
import UTLutils as ut
import UTLprepro as utprepro
import UTLcrossings as utxings
import UTLmag as utmag
import MODionosphere as modionos
import os.path
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, splev, splrep


def Get_ElectroDynamic(Juls_HI, Jxs, SigP_hi, SigH_hi, Inc_deg, NS, TT_X, TT_Y, theta, count4):
    # +
    # =================================================
    # Description: Subroutine to calculate the ionospheric parameters.
    # Input:
    #    Juls_HI: julday
    #    Jxs: A/m
    #    SigP_hi: mho
    #    SigH_hi: mho
    #    Inc_deg: inclination angle
    #    NS: 0-North, 1-South
    #    TT_X, TT_Y: xy coordinates of footprints
    #    theta: angle between the trajectories and the direction perpendicular to the main oval
    #    count4: N_elements of time series
    #    Jys: A/m
    #    Exs: V/m
    #    Jouleheating: W/m2
    #    Phix: V
    # Output:
    # Example:
    # History:
    # =================================================
    # -
    Rj = 71492.0e3  # m

    Exs = np.zeros(len(SigP_hi))
    Exs = Jxs / SigP_hi * np.sin(ut.deg2rad(Inc_deg))**2   # V/m
    # A/m   # signed Inc_deg, + for downward
    Jys = - SigH_hi*Exs / np.sin(ut.deg2rad(Inc_deg))
    Jouleheating = Jxs*Exs   # W/m2

    Phix = Juls_HI*0  # V
    if NS == 0:
        # [North side, in to out.]
        i = count4 - 2
        while i >= 0:
            dxy = np.sqrt((TT_X[i] - TT_X[i+1])**2 +
                          (TT_Y[i] - TT_Y[i+1])**2)  # Rj
            Ex0 = (Exs[i] + Exs[i+1])/2.0  # V/m
            Phix[i] = Phix[i+1] + dxy*Ex0*Rj*np.cos(ut.deg2rad(theta))
            i = i - 1
    else:
        # [For south: outside - in.]
        for i in range(1, count4):
            dxy = np.sqrt((TT_X[i] - TT_X[i-1])**2 +
                          (TT_Y[i] - TT_Y[i-1])**2)  # Rj
            Ex0 = (Exs[i] + Exs[i-1])/2.0  # V/m
            Phix[i] = Phix[i-1] - dxy*Ex0*Rj*np.cos(ut.deg2rad(theta))

    return Jys, Exs, Jouleheating, Phix


def Jupiter_Electrodynamics_Model(PJnum, NS, NC):
    # ------------------------------
    # Step 1/2: Set the time
    # ------------------------------
    # time range: see hyperparameters.py.
    tlim = utxings.Get_MainOvalCrossUTs(
        PJnum, NS, NC, OffSet_min=hp.Offset_min)
    # ------------------------------
    # Step 2/2: Set the time range Juno
    #   in and out of the auroral regions.
    # ------------------------------

    # [T0-T1 as the time interval Juno passes the main oval.]
    # [North: inside - out, South: outside - in]
    T0 = tlim[0]
    jul0 = ut.yxutc2jul(T0)

    T1 = tlim[1]
    jul1 = ut.yxutc2jul(T1)

    # [time series from tlim with higher resolution: e.g., 2s]
    julim = [ut.yxutc2jul(T0), ut.yxutc2jul(T1)]
    # see resolution in hyperparameters.py
    Nsamp = (julim[1] - julim[0])*86400.0/(1.0*hp.resolution)
    Juls_HI = np.linspace(julim[0], julim[1], int(np.round(Nsamp)))
    Juls_HI = Juls_HI[:-1]

    # ==================Main Program Begin================================
    # (1) Load Ephemeris data: 60s resolution -> 2s

    filep = hp.pathtodata + 'ephemeris/'

    filen = 'Ephem_Juno_jrm09_can_perijove' + \
        str(PJnum).zfill(2) + '_step=60sec.sav'

    savfile = sp.io.readsav(filep + filen)
    JULs = savfile.juls
    UT = savfile.ut
    R_FP = savfile.r_fp
    TH_FP = savfile.th_fp
    PHI_FP = savfile.phi_fp
    R_SC = savfile.r_sc
    LAT_IAU = savfile.lat_iau
    LON_IAU = savfile.lon_iau

    # [data clipping]
    tlim_jul = [ut.yxutc2jul(tlim[0]), ut.yxutc2jul(tlim[1])]
    eps = (JULs[1] - JULs[0])/10
    ind = np.where((JULs >= tlim_jul[0] - eps)
                   & (JULs <= tlim_jul[1] + eps))[0]
    count = len(ind)
    if count <= 0:
        raise ValueError('Error: L49, time of interest is out of range!')
    UT_toi = UT[ind]
    juls_toi = JULs[ind]
    R_FP_toi = R_FP[ind]
    TH_FP_toi = TH_FP[ind]
    PHI_FP_toi = PHI_FP[ind]
    R_IAU_toi = R_SC[ind]
    LAT_IAU_toi = LAT_IAU[ind]
    LON_IAU_toi = LON_IAU[ind]

    # [Cartesian coords on the surface: x, y is NOT necessary para/perp to the oval]
    # [To calculate the distance for each 2s]
    TT_X = R_FP_toi*np.sin(TH_FP_toi*np.pi/180) * \
        np.cos(PHI_FP_toi*np.pi/180)  # Footprints position
    TT_Y = R_FP_toi*np.sin(TH_FP_toi*np.pi/180) * \
        np.sin(PHI_FP_toi*np.pi/180)  # in Rj

    # [Interpolation to high resolution]
    # Footprints Position

    f_TT_X = interp1d(juls_toi, TT_X, fill_value="extrapolate")
    TT_X = f_TT_X(Juls_HI)   # Rj
    f_TT_Y = interp1d(juls_toi, TT_Y, fill_value="extrapolate")
    TT_Y = f_TT_Y(Juls_HI)  # Rj
    f_R_FP_toi = interp1d(juls_toi, R_FP_toi, fill_value="extrapolate")
    R_FP_toi = f_R_FP_toi(Juls_HI)
    f_TH_FP_toi = interp1d(juls_toi, TH_FP_toi, fill_value="extrapolate")
    TH_FP_toi = f_TH_FP_toi(Juls_HI)
    f_PHI_FP_toi = interp1d(juls_toi, PHI_FP_toi, fill_value="extrapolate")
    PHI_FP_toi = f_PHI_FP_toi(Juls_HI)

    # Spacecraft Position
    f_R_IAU_toi = interp1d(juls_toi, R_IAU_toi, fill_value="extrapolate")
    R_IAU_toi = f_R_IAU_toi(Juls_HI)
    f_TH_IAU_toi = interp1d(juls_toi, LAT_IAU_toi, fill_value="extrapolate")
    TH_IAU_toi = 90.0 - f_TH_IAU_toi(Juls_HI)
    f_PHI_IAU_toi = interp1d(juls_toi, LON_IAU_toi, fill_value="extrapolate")
    PHI_IAU_toi = f_PHI_IAU_toi(Juls_HI)
    UTs_HI = np.array([ut.yxjul2utc(jul) for jul in Juls_HI])

    # (2) Calculate the FACs: (+ for upward)
    # (2.1) Load dB data: in JRM09 coords.

    filename = "dB_" + ut.get_name(PJnum, NS, NC) + ".npz"
    if not os.path.isfile(hp.pathtoresB + filename):
        print("No mag data file found, calling prepro.mag().")
        dtarr, dBr_nT, dBt_nT, dBp_nT, Bz, Bmag = utprepro.mag(PJnum, NS, NC)
        print("Mag data generated, going back to electrodynamics.py.")
        Juls_Bp = ut.datetime2julian(dtarr)
    else:
        dBfile = np.load(hp.pathtoresB + filename)
        Juls_Bp = dBfile['Juls_Bp']
        dBr_nT = dBfile['dBr_nT']
        dBt_nT = dBfile['dBt_nT']
        dBp_nT = dBfile['dBp_nT']
        dtarr = ut.julian2datetime(Juls_Bp)

    # spline interpolation
    tck_dBr = splrep(Juls_Bp, dBr_nT)
    dBr_nT = splev(Juls_HI, tck_dBr)  # interp(dBp_nT, Juls_Bp, Juls_HI)

    tck_dBt = splrep(Juls_Bp, dBt_nT)
    dBt_nT = splev(Juls_HI, tck_dBt)

    tck_dBp = splrep(Juls_Bp, dBp_nT)
    dBp_nT = splev(Juls_HI, tck_dBp)

    # (2.2) from SymIII to JRM09
    # [Juno Locations in JRM09 coordinate]
    jno_xyz = ut.Get_JunoState(UTs_HI, 'iau')
    jno_rtp = ut.Car2SphP(jno_xyz[:, 0], jno_xyz[:, 1], jno_xyz[:, 2])
    jno_xyz_iau = jno_xyz
    jno_rtp_iau = jno_rtp

    Posixyz = utmag.Coord_SIII2JRM09(np.transpose(jno_xyz))   # SymIII to JRM09
    Posirtp = ut.Car2SphP(
        Posixyz[0, :], Posixyz[1, :], Posixyz[2, :])  # Cart -> Sphr
    jno_xyz_jrm = Posixyz
    jno_rtp_jrm = Posirtp

    R_SC_JRM = Posirtp[:, 0]
    TH_SC_JRM = Posirtp[:, 1]

    # [Juno footpoints in JRM09 coordinate]
    Posixyz = np.transpose(ut.Sph2CarP(R_FP_toi, TH_FP_toi, PHI_FP_toi))
    P_XYZ_FP_IAU = Posixyz
    Posixyz = utmag.Coord_SIII2JRM09(Posixyz)
    Posirtp = ut.Car2SphP(Posixyz[0, :], Posixyz[1, :], Posixyz[2, :])
    P_RTP_FP_JRM = Posirtp
    TH_FP_JRM = Posirtp[:, 1]

    # (2.3) Calculate the FACs at footprints
    # [Juno B]
    Br_Juno, Bt_Juno, Bp_Juno, Bmag_Juno = utmag.GetJupiterMag(
        R_IAU_toi, TH_IAU_toi, PHI_IAU_toi, CAN=1)
    # [Footprint B]
    Br_JuFP, Bt_JuFP, Bp_JuFP, Bmag_JuFP = utmag.GetJupiterMag(
        R_FP_toi, TH_FP_toi, PHI_FP_toi, CAN=1)

    Rj = 71492.0e3
    ddBp = ut.yxdiff(dBp_nT)
    ddtht = ut.yxdiff(np.pi/180*(TH_SC_JRM))
    ddBp = np.append(ddBp[0], ddBp)
    ddtht = np.append(ddtht[0], ddtht)
    Jpara = Bmag_JuFP / Bmag_Juno / (4*np.pi*1.0e-7) * (1.0/R_IAU_toi/Rj*ddBp/ddtht + \
            dBp_nT/R_IAU_toi/Rj*np.cos(np.pi*TH_SC_JRM/180)/np.sin(np.pi*TH_SC_JRM/180))
    Jpara *= 1.0e-9  # A/m2

    # [Calculate Inclination angle to horizon: positive for downward: - for north, + for south]
    BR = Br_JuFP
    BT = np.sqrt(Bt_JuFP*Bt_JuFP+Bp_JuFP*Bp_JuFP)   # tangent B
    Inc_deg = - 180/np.pi * np.arctan2(BR, BT)    # 北半球为负，南半球为正, 体现在Br上

    # Sariah on Friday 14.01.2022
    # New calculation of Jx, previously was given by :
    # # (3) Derive Jx (A/m)
    # Jx = - R_IAU_toi*np.sin(np.pi*TH_SC_JRM/180) * dBp_nT / (4*np.pi*1.0e-7)/np.sin(np.pi*TH_FP_JRM/180)/R_FP_toi*1.0e-9 # Model B: Spherical model. A/m

    Jx = np.zeros(len(dBp_nT))
    st_pers = utxings.limits_ThetaFP(PJnum, NS, NC)
    for i in range(len(st_pers)):
        (t_FP_min, t_FP_max) = st_pers[i]
        study_period = np.where((TH_FP_JRM > t_FP_min)
                                & (TH_FP_JRM < t_FP_max))[0]
        integrand_Jx = 0
        for i in study_period:
            integrand_Jx += Jpara[i]*np.abs(np.sin(Inc_deg[i]*np.pi/180))*np.sin(
                np.pi*TH_SC_JRM[i]/180)*(np.pi/180)*(TH_SC_JRM[i] - TH_SC_JRM[i-1])
            Jx[i] = - integrand_Jx * R_FP_toi[i] * \
                Rj / np.sin(np.pi*TH_SC_JRM[i]/180)

        # enforce boudary conditions at equatorial side
        # if (NS == 0)&(i == len(st_pers) - 1) or (NS == 1)&(i == 0):
        #     d_0 = np.abs(TH_FP_JRM[study_period[0]] - 90)
        #     d_m1 = np.abs(TH_FP_JRM[study_period[-1]] - 90)
        #     if d_0 < d_m1:
        #         Jx[study_period] -= Jx[study_period[-1]]

    ind4 = np.where((Juls_HI >= jul0) & (Juls_HI <= jul1))[0]
    count4 = len(ind4)
    if count4 <= 0:
        raise ValueError('Error: main oval time is outside of range!')

    # now theta is the angle to x
    theta = 90 - utxings.Get_ThetaTrajectory(PJnum, NS, NC)

    # (4) Load Pedersen/Hall conductance
    filen = 'Cond_' + ut.get_name(PJnum, NS, NC) + '.npz'
    if not os.path.isfile(hp.pathtoiono + filen):
        print("Conductance file does not exist, calling MODionos.py to generate it.")
        modionos.Sub_Cond_AlongJuno(PJnum, NS, NC)
        print("leaving MODionos.py, going back to electrodynamics.py.")

    npzfile = np.load(hp.pathtoiono + filen)
    JULS = npzfile['juls']
    PARAMALL = npzfile['ParamAll']
    Qs = npzfile['Qs']
    TEFlux = npzfile['TEFlux']

    Juls_p = JULS
    SigP = PARAMALL[0]
    SigH = PARAMALL[1]
    SigP_CH4 = PARAMALL[3]
    SigH_CH4 = PARAMALL[4]
    nanSigP = np.where(np.isnan(SigP))[0]
    notnanSigP = np.where(~np.isnan(SigP))[0]
    for i in nanSigP:
        id = notnanSigP[notnanSigP < i]
        lend = len(id)
        if lend == 0:
            id = 0
        else:
            id = id[-1]
        iu = notnanSigP[notnanSigP > i]
        lenu = len(iu)
        if lenu == 0:
            iu = lenu
        else:
            iu = iu[0] + 1
        SigP[i] = np.nanmean(SigP[id: iu])
        SigH[i] = np.nanmean(SigH[id: iu])
        SigP_CH4[i] = np.nanmean(SigP_CH4[id: iu])
        SigH_CH4[i] = np.nanmean(SigH_CH4[id: iu])
        TEFlux[i] = np.nanmean(TEFlux[id: iu])

    # Note here Juls_HI is already filtered as inside the oval
    f_SigP_hi = interp1d(Juls_p, SigP, fill_value="extrapolate")
    SigP_hi = f_SigP_hi(Juls_HI)  # mho
    f_SigH_hi = interp1d(Juls_p, SigH, fill_value="extrapolate")
    SigH_hi = f_SigH_hi(Juls_HI)  # mho
    f_SigP_CH4_hi = interp1d(Juls_p, SigP_CH4, fill_value="extrapolate")
    SigP_CH4_hi = f_SigP_CH4_hi(Juls_HI)  # mho
    f_SigH_CH4_hi = interp1d(Juls_p, SigH_CH4, fill_value="extrapolate")
    SigH_CH4_hi = f_SigH_CH4_hi(Juls_HI)  # mho
    f_QS_hi = interp1d(Juls_p, Qs, fill_value="extrapolate")
    QS_hi = f_QS_hi(Juls_HI)  # mho

    # (5) Calculate Ey, Jx, and Joule heating
    # [without CH4]
    Jxs = Jx  # A/m
    Jys, Exs, Jouleheating, Phix = Get_ElectroDynamic(
        Juls_HI, Jxs, SigP_hi, SigH_hi, Inc_deg, NS, TT_X, TT_Y, theta, count4)

    # [with CH4]
    Jys_CH4, Exs_CH4, Jouleheating_CH4, Phix_CH4 = Get_ElectroDynamic(
        Juls_HI, Jxs, SigP_CH4_hi, SigH_CH4_hi, Inc_deg, NS, TT_X, TT_Y, theta, count4)

    # (6) Calculate Subcorotation
    # [JRM09]
    Etcot = np.pi*2.0 / (9.9259*3600.0)*R_FP_toi*71492.0e3*np.sin(np.pi * \
            TH_FP_JRM/180)*Bmag_JuFP*1.0e-9*np.sin(np.pi*Inc_deg/180)
    Et = Exs

    KK = hp.KK
    # Omega_plasma = (1.0 - dOmega) [Omega_J]
    dOmega = np.abs(Et / Etcot / (1.0 - KK))

    omega_Jup = np.pi*2.0 / (9.9259*3600.0)  # Omega_J, rad/s
    omega_plasma = (1.0 - dOmega)*omega_Jup  # in unit of rad/s

    # rho, distance to rotation axis
    rho_i = R_FP_toi*71492.0e3*np.sin(np.pi*TH_FP_JRM/180)
    B_i = Bmag_JuFP*1.0e-9*np.sin(np.pi*Inc_deg/180)   # B in T, at footprint

    # Power in W/m2
    Power_total = np.abs(omega_Jup*rho_i*B_i*Jxs)   # total power
    Power_M = np.abs(omega_plasma*rho_i*B_i*Jxs)    # to magnetosphere
    Power_A = np.abs(dOmega*omega_Jup*rho_i*B_i*Jxs)  # atmosphere heating
    Power_J = np.abs(Exs*Jxs)  # Joule heating
    Power_D = Power_A - Power_J    # ion drag

    # [Save variables]
    Jxys = np.array([Jxs, Jys, Exs, Jouleheating, Phix])
    Jxys_CH4 = np.array([Jxs, Jys_CH4, Exs_CH4, Jouleheating_CH4, Phix_CH4])

    Power = np.array([TH_FP_JRM, Power_total, Power_M,
                     Power_A, Power_J, Power_D])

    filename = 'JouleHeating_' + ut.get_name(PJnum, NS, NC)
    np.savez(hp.pathtoelec + filename, T0=T0, T1=T1, Juls_HI=Juls_HI, Jpara=Jpara, Jxys=Jxys, Jxys_CH4=Jxys_CH4,
             dOmega=dOmega, QS_hi=QS_hi, Power=Power, SIGP=SigP, SIGH=SigH, SIGP3=SigP_CH4, SIGH3=SigH_CH4, TEFlux=TEFlux, JULS=JULS)
    # This file is calculated Joule heating
    # including the T0-1 the period of main oval crossing,
    # Jpara from Axis-symmetric model,
    # Jpara_JADE from particle data in T
    # Jxys = [Jx, Jy, Ex, Jouleheating, PhiX],
    # Jxys_CH4 same but with conductances including CH4.
    # 1 - dOmega is the plasma omega in inertial frame.
    # Power is in W/m2
    # [TH_FP_JRM, Power_total, Power_M: to Magneto, Power_A: heat in atmos,
    # Power_J: Jouleheating, Power_D: ion drag.]
    print('finished!')
    return T0, T1, Juls_HI, Jpara, Jxys, Jxys_CH4, dOmega, QS_hi, Power, SigP, SigH, SigP_CH4, SigH_CH4, TEFlux, JULS


if __name__ == "__main__":
    Jupiter_Electrodynamics_Model(1, 0, 0)