import UTLutils as ut
import UTLprepro as prepro
import UTLcrossings as utxings
import MODionosphere as ionos
import MODelectrodynamics as ele
import hyperparameters as hp
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as dates
from datetime import datetime, timedelta
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d

NSstr = hp.NSlong
NSstr2 = hp.NS



def CaseStudies(PJnum, NS, NC, show=False, useFP=False):
    tlim = utxings.Get_CrossingUTs(PJnum, NS, NC)
    name = "PJ" + str(PJnum).zfill(2) + NSstr2[NS] + str(NC)
    if tlim[0] == 0 or tlim[0] == '0' :
        print("No crossing named", name + '.')
        # print(tlim)
        return 0
    else:
        print("Crossing", name, 'found !')
        dtstart = datetime.fromisoformat(tlim[0])
        dtend = datetime.fromisoformat(tlim[1])

    filename = 'JouleHeating_' + ut.get_name(PJnum, NS, NC) + ".npz"
    
    if not os.path.isfile(hp.pathtoelec + filename):
        print("Results file does not exist, calling electrodynamics.py to generate it.")
        T0, T1, Juls_HI, Jpara, Jxys, Jxys_CH4, dOmega, QS_hi, Power, SIGP, SIGH, SIGP3, SIGH3, TEFlux, JULS = ele.Jupiter_Electrodynamics_Model(PJnum, NS, NC)
        print("Leaving electrodynamics.py, going back to epsc.py.")
    else: 
        npzfile = np.load(hp.pathtoelec + filename)
        T0 = npzfile["T0"]
        T1 = npzfile["T1"]
        Juls_HI = npzfile["Juls_HI"]
        Jpara = npzfile["Jpara"]
        Jxys = npzfile["Jxys"]
        Jxys_CH4 = npzfile["Jxys_CH4"]
        dOmega = npzfile["dOmega"]
        QS_hi = npzfile["QS_hi"]
        Power = npzfile["Power"]
        # This file is calculated Joule heating
        # including the T0-1 the period of main oval crossing, 
        # Jpara from Axis-symmetric model, 
        # Jxys = [Jx, Jy, Ex, Jouleheating, PhiX],
        # Jxys_CH4 same but with conductances including CH4.
        # 1 - dOmega is the plasma omega in inertial frame.
        # Power = [
        #       TH_FP_JRM,
        #       Power_total,
        #       Power_M: to Magneto, 
        #       Power_A: heat in atmos,
        #       Power_J: Jouleheating, 
        #       Power_D: ion drag.
        #   ] in W/m2
        JULS = npzfile['JULS']              # unit: julday
        TEFlux = npzfile['TEFlux']          # unit: W/m2
        SIGP = npzfile['SIGP']          # unit: mho
        SIGH = npzfile['SIGH']          # unit: mho
        SIGP3 = npzfile['SIGP3']          # unit: mho
        SIGH3 = npzfile['SIGH3']          # unit: mho
        # This is the conductances calculated based on Taos model. 
        # units are: mho, X1.0e12 /cm2, W/m2, julday. 
        # ParamAll = [SIGP, SIGH, N_ele_col, SIGP3, SIGH3, N_ele_col3, N_H3p_col3]
        # SIGP is for without CH4, and SIGP3 is for with CH4 included
    # adapting units to plots
    dtarr = ut.julian2datetime(Juls_HI)
    Jxys = Jxys*1000        # from W/m2 to mW/m2
    Jxys_CH4 = Jxys_CH4*1000
    Jxys[4] /= 1e6
    Jxys_CH4[4] /= 1e6
    ThetaFP = Power[0]
    # Jxys = [Jx, Jy, Ex, Jouleheating, PhiX]
    # PhiX is alredy in V, it got converted into mV, so we divide it 
    # by 1e6 to convert it to kV

    TEFlux = TEFlux*1000 #from W/m2 to mW/m2
    dtarr1 = ut.julian2datetime(JULS)

    time_common, e0_kev, pflux, pflux_jade, pflux_jedi = prepro.eflux(PJnum, NS, NC)
    # pflux is in c/s/sr/cm2/eV
    time_common_up, e0_kev_up, pflux_up, pflux_jade_up, pflux_jedi_up = prepro.eflux(PJnum, NS, NC)
    dtarr_up, teflux_up, tflux_up = ionos.totalFlux(PJnum, NS, NC)
    teflux_up = teflux_up*1000 # from W/m2 to mW/m2
    # time_up, e0_kev_up, pflux_up, pflux_jade_up, pflux_jedi_up = prepro.eflux(PJnum, NS, NC, up=True)
    # get dB data
    filenamedB = "dB_" + name + ".npz"
    dBdata = np.load(hp.pathtoresB + filenamedB)
    Juls_Bp = dBdata['Juls_Bp']
    dBt_nT = dBdata['dBt_nT']
    dBp_nT = dBdata['dBp_nT']
    dtarrdB = ut.julian2datetime(Juls_Bp)
    Br = dBdata['Br']
    Bmag = dBdata['Bmag']
    vDrift = ut.Get_DriftVelocity(Juls_Bp, Br, Bmag, Juls_HI, Jxys[2])
    vDrift_CH4 = ut.Get_DriftVelocity(Juls_Bp, Br, Bmag, Juls_HI, Jxys_CH4[2])

    timeUVS, brightnessUVS = prepro.get_UVS_profile(PJnum, NS, NC)

    # then plots the data
    fig, ax = plt.subplots(10, 1, sharex=True, figsize=(7.0, 9.0))
    ax[4].set_visible(False)
    # fig, ax = plt.subplots(5, 1, sharex=True, figsize=(4.0, 4.5))
    
    time_common_bis = np.copy(time_common)
    dtarr1_bis = np.copy(dtarr1)
    dtarrdB_bis = np.copy(dtarrdB)
    time_common_up_bis = np.copy(time_common_up)
    dtarr_up_bis = np.copy(dtarr_up)
    dtarr_bis = np.copy(dtarr)
    dtstart_bis = dtstart
    dtend_bis = dtend
    timeUVSbis = np.copy(timeUVS)

    Jxys[0][Jxys[0]==0]=np.nan
    vDrift[vDrift==0] = np.nan
    Jxys[3][Jxys[0]==0] = np.nan
    if (PJnum == 5)&(NS==1):
        Jxys[3][(ThetaFP>165.3)&(ThetaFP<166)] = np.nan
        thetadB = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarrdB)
        vDrift[(thetadB>165.3)&(thetadB<166)] = np.nan
    
    # compute difference time UVS/data
    ind1 = ((dtarr1 >= dtstart) & (dtarr1 <= dtend) & (dtarr1_bis >= dtstart_bis) & (dtarr1_bis <= dtend_bis))
    indUVS = ((timeUVS >= dtstart) & (timeUVS <= dtend) & (timeUVSbis >= dtstart_bis) & (timeUVSbis <= dtend_bis) & (brightnessUVS>-500))
    TEFlux_interpol = interp1d([time.timestamp() for time in dtarr1[ind1]], TEFlux[ind1], fill_value="extrapolate")
    TEFlux_plot= TEFlux_interpol([time.timestamp() for time in timeUVS[indUVS]])
    timeseries=timeUVS[indUVS]
    cov=0
    if name == 'PJ03N0':
        delay = timedelta(seconds=80)
    elif name == 'PJ22N0':
        delay = timedelta(seconds=59)
    elif name == 'PJ12N0':
        delay = timedelta(seconds=50)
    elif name == 'PJ05S3':
        delay = timedelta(seconds=56)
    elif name == 'PJ22N3':
        delay = timedelta(seconds=68)
    elif len(timeseries) != 0:
        brightnessUVS_plot=brightnessUVS[indUVS]
        k=0
        covariance=0
        DeltaTime=0
        while k < int(len(timeseries)/2):
            new_covariance=np.sum(TEFlux_plot[0:len(timeseries)-k]*brightnessUVS_plot[k:len(timeseries)])/(len(timeseries)-k+1)
            #print(new_covariance)
            if new_covariance > covariance :
                covariance = new_covariance
                DeltaTime = k
            k=k+1
        delay = timeseries[DeltaTime]-timeseries[0]
    else: 
        delay = timedelta(seconds=0)


    timeUVS = timeUVS - delay
    stH = dtstart.strftime('%Y/%m/%d - %H:%M') + ' - ' + dtend.strftime('%H:%M')
    dtarr_original = np.copy(dtarr)
    if useFP:
        # Here, lists of datetime are converted to lists of angles
        time_common = ut.datetimeToThetaFP(dtarr, ThetaFP, time_common)
        dtarr1 = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarr1)
        dtarrdB = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarrdB)
        time_common_up = ut.datetimeToThetaFP(dtarr, ThetaFP, time_common_up)
        dtarr_up = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarr_up)
        timeUVS = ut.datetimeToThetaFP(dtarr, ThetaFP, timeUVS)
        dtarr = ThetaFP

        if NS:
            dtstart = max(np.min(ThetaFP[(dtarr_bis >= dtstart_bis) & (dtarr_bis <= dtend_bis)]), 145)
            dtend = min(np.max(ThetaFP[(dtarr_bis >= dtstart_bis) & (dtarr_bis <= dtend_bis)]), 185)
        else:
            dtstart = max(np.min(ThetaFP[(dtarr_bis >= dtstart_bis) & (dtarr_bis <= dtend_bis)]), 5)
            dtend = min(np.max(ThetaFP[(dtarr_bis >= dtstart_bis) & (dtarr_bis <= dtend_bis)]), 40)
        ax[9].set_xlabel("Colatitude of FootPrint (deg)")
    else:
        ax[9].set_xlabel("Time")


    # ax[5]: Conductances
    lwidth = 1
    ax[5].plot(dtarr1[ind1], SIGH[ind1], linewidth=lwidth, label=r'$\Sigma_H$')
    ax[5].plot(dtarr1[ind1], SIGP[ind1], linewidth=lwidth, label=r'$\Sigma_P$')
    ax[5].set_ylabel(r'$\Sigma_{(\Omega^{-1})}$')
    ax[5].legend(loc='center right')
    # ax[7]: fac
    ind2 = ((dtarrdB >= dtstart) & (dtarrdB <= dtend) & (dtarrdB_bis >= dtstart_bis) & (dtarrdB_bis <= dtend_bis))
    ind5 = ((dtarr >= dtstart) & (dtarr <= dtend) & (dtarr_bis >= dtstart_bis) & (dtarr_bis <= dtend_bis))
    ax[6].plot(dtarr[ind5], Jpara[ind5]*1e6, 'C0', linewidth=lwidth)
    ax[6].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[6].set_ylabel(r"$FAC_{(\mu A/m^2)}$")
    # ax[7]: Jx, y


    ## /!\ Attention, traitement séparé de Nord et Sud  pour J_x !!!
    if NS == 0:
        ax[7].plot(dtarr[ind5], Jxys[0][ind5], 'C1', linewidth=lwidth)
    else:
        ax[7].plot(dtarr[ind5], -Jxys[0][ind5], 'C1', linewidth=lwidth)
    ## /!\ Attention, fin de traitement séparé de Nord et Sud !!!

    
    ax[7].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[7].set_ylabel(r'$J_{x, (mA/m)}$')
    # ax[8]: ExB
    Vd = vDrift[ind2]
    ax[8].plot(dtarrdB[ind2], Vd, 'C0', linewidth=lwidth)
    ax[8].set_ylabel(r"$(E \times B)_{y, (m/s)}$")
    ax[8].axhline(y=0, color='k', ls='--', lw=0.5)

    # ax[9]: Joule Heating: Pj, Pele
    # Power in W/m2, necessary to convert it to mW/m2
    ax[9].plot(dtarr1[ind1], TEFlux[ind1], 'C1', linewidth=lwidth, label=r'$P_e$')
    ax[9].plot(dtarr[ind5], Jxys[3][ind5], 'C0', linewidth=lwidth, label=r'$P_J$')
    ax[9].legend(loc='center right')
    ax[9].set_ylabel(r'$P_{(mW/m^2)}$')
    ax[9].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[5].set_xlim(dtstart, dtend)
    ax[3].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0].annotate('(a)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[1].annotate('(b)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[2].annotate('(c)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[3].annotate('(d)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[5].annotate('(e)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[6].annotate('(f)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[6].annotate('up', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.75))
    ax[6].annotate('down', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.2))
    ax[7].annotate('(g)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[7].annotate('equator', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.75))
    ax[7].annotate('pole', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.2))
    ax[8].annotate('(h)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[8].annotate('east', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.75))
    ax[8].annotate('west', xy=(0,0), xycoords='axes fraction', xytext=(1.01, 0.2))
    ax[9].annotate('(i)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))

    for i in range(5, 10):
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())

    io = utxings.Time_Crossing_Io(PJnum, NS, NC)
    if io != 0:
        tlio = io[0]
        tuio = io[1]
        taio = tlio + (tuio - tlio)/10
        indulio = ((dtarr_bis > tlio) & (dtarr_bis < tuio))
        indaio = ((dtarr_bis > taio) & (dtarr_bis < tuio))
        taio = dtarr[indaio][0]
        tlio = dtarr[indulio][0]
        tuio = dtarr[indulio][-1]
        ax[9].annotate('Io', xycoords='data', xy = ((taio, 200)))
        for i in range(5, 10):
            ax[i].axvspan(tlio, tuio, alpha = 0.15, color = "C1")

    fig.align_ylabels()

    ### Plot instrument data

    cbar_ax1 = fig.add_axes([0.91, 0.81, 0.01, 0.07]) # tight layout
    # ax[0]: eflux downward
    pflux = np.log10(pflux)
    pcm1 = ax[0].pcolormesh(
        time_common[(time_common_bis >= dtstart_bis) & (time_common_bis <= dtend_bis)],
        e0_kev,
        pflux[:,(time_common_bis >= dtstart_bis) & (time_common_bis <= dtend_bis)],
        cmap = 'Spectral_r',
        norm=colors.Normalize(vmin=1, vmax=6),
        shading='gouraud'
        )
    cbar1 = fig.colorbar(pcm1, ax=ax[0], cax=cbar_ax1, extend='both')
    cbar1.set_label(r'$Flux_{/s/sr/cm2/eV}$')
    ax[0].set_ylabel(r'$E_{(keV)}$')
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-1, 1e3)
    ax[0].set_xlim(dtstart, dtend)
    ax[0].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[0].plot(dtarr1[ind1], np.abs(TEFlux[ind1]), 'k', linewidth=lwidth)
    # ax[1].plot(dtarrdB[ind2], dBt_nT[ind2], 'C1', label=r'$\delta B_{\theta}$', linewidth=lwidth)
    ax[1].plot(dtarrdB[ind2], dBp_nT[ind2], 'k', label=r'$\delta B_{\phi}$', linewidth=lwidth-0.2)
    ax[1].set_ylabel(r'$\delta B_{(nT)}$')
    ax[1].legend(loc = 'center right')
    ax[1].axhline(y=0, color='k', ls='--', lw=0.5)
    indUVS = ((timeUVS >= dtstart) & (timeUVS <= dtend) & (timeUVSbis >= dtstart_bis) & (timeUVSbis <= dtend_bis) & (brightnessUVS>-500))
    if name == "PJ05S3":
        indUVS = ((timeUVS >= dtstart) & (timeUVS <= dtend) & (timeUVSbis >= dtstart_bis) & (timeUVSbis <= dtend_bis) & (brightnessUVS>-500)&(brightnessUVS<50))
    # ax[3].scatter(timeUVS[indUVS], brightnessUVS[indUVS], s=10)
    ax[3].plot(timeUVS[indUVS], brightnessUVS[indUVS])
    ax[3].set_ylabel(r"br$($H2$)_{(kR)}$")


    wavelim = utxings.Get_yLim_Waves(PJnum, NS, NC)
    ax[2].set_ylim(wavelim[0], wavelim[1])
    
    for i in range(1,4):
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())

    ax[0].title.set_text("Crossing " + name[:5] + " - " + stH + "\nJuno instrument data")
    ax[5].title.set_text("Calculated MIT coupling key parameters")
    if useFP:
        ax[9].xaxis.set_minor_locator(AutoMinorLocator(n=10))
    else:
        ax[9].xaxis.set_minor_locator(AutoMinorLocator(n=6))
    #plt.tight_layout()
    if not useFP:
        ax[9].xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))


    if name == "PJ06S0" and useFP:
        for i in range(5, 10):
            ax[i].axvspan(163.9, 165, alpha = 0.1, color = "C0")
            ax[i].axvspan(166.4, 167.5, alpha = 0.1, color = "C1")

    if show:
        plt.show()
    else:
        if useFP:
            plt.savefig(hp.pathtoplots + 'colatitude/c' + name + '.png', bbox_inches="tight")
        else:
            plt.savefig(hp.pathtoplots + 'time/t' + name + '.png', bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    show = False
    pj, ns, nc = 1, 0, 0
    CaseStudies(pj, ns, nc, show=show, useFP=True)