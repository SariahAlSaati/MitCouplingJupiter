import UTLutils as ut
import UTLprepro as prepro
import UTLcrossings as utxings
import MODionosphere as ionos
import MODelectrodynamics as ele
import hyperparameters as hp
import os.path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d


def get_data(pj, ns, nc):
    use_ancient = True
    tlim = utxings.Get_CrossingUTs(pj, ns, nc)
    name = ut.get_name(pj, ns, nc)
    if tlim[0] == 0 or tlim[0] == '0' :
        return 0

    filename = 'JouleHeating_' + name + ".npz"
    if not os.path.isfile(hp.pathtoelec + filename):
        print("Results file does not exist, calling MODelectrodynamics.py to generate it.")
        T0, T1, Juls_HI, Jpara, Jxys, Jxys_CH4, dOmega, QS_hi, Power, SIGP, SIGH, SIGP3, SIGH3, TEFlux, JULS = ele.Jupiter_Electrodynamics_Model(pj, ns, nc)
        print("Leaving MODelectrodynamics.py, going back to PLTsuperposed.py.")
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
        JULS = npzfile['JULS']              # unit: julday
        TEFlux = npzfile['TEFlux']          # unit: W/m2
        SIGP = npzfile['SIGP']          # unit: mho
        SIGH = npzfile['SIGH']          # unit: mho
        SIGP3 = npzfile['SIGP3']          # unit: mho
        SIGH3 = npzfile['SIGH3']
    dtarr = ut.julian2datetime(Juls_HI)
    Jxys = Jxys*1000        # from W/m2 to mW/m2
    Jxys_CH4 = Jxys_CH4*1000
    Jxys[4] /= 1e6
    Jxys_CH4[4] /= 1e6
    ThetaFP = Power[0]
    TEFlux = TEFlux*1000 #from W/m2 to mW/m2
    dtarr1 = ut.julian2datetime(JULS)
    time_common, e0_kev, pflux, pflux_jade, pflux_jedi = prepro.eflux(pj, ns, nc)
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

    if use_ancient:
        positionMainOval = utxings.Get_MainOvalLongitude(pj, ns)
    else:
        dtXing = utxings.Time_MainOval(pj, ns, nc)
        positionMainOval= ThetaFP[dtarr>=dtXing][0]

    # Here, lists of datetime are converted to lists of angles
    theta_common = ut.datetimeToThetaFP(dtarr, ThetaFP, time_common) - positionMainOval
    theta1 = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarr1) - positionMainOval
    thetadB = ut.datetimeToThetaFP(dtarr, ThetaFP, dtarrdB) - positionMainOval
    theta = ThetaFP - positionMainOval

    Jxys[0][Jxys[0]==0]=np.nan
    vDrift[vDrift==0] = np.nan
    Jxys[3][Jxys[0]==0] = np.nan
    if pj == 5 and ns == 1:
        Jxys[3][(theta>-0.7)&(theta<0)]=np.nan
        vDrift[(thetadB>-0.7)&(thetadB<0)] = np.nan


    return theta1, dtarr1, thetadB, dtarrdB, theta, dtarr, SIGH, SIGP, Jpara*1e6, Jxys, vDrift, TEFlux



def plotSuperposed(NS, show=False):
    if NS == 0:
        thetastart = -2.0
        thetaend = 2.0
    else: 
        thetastart = -1.7
        thetaend = 1.7
    if NS == 0:
        PJlist = [1, 3, 7, 9, 11, 12, 13, 14, 21, 22] 
    else:
        PJlist = [3, 4, 5, 6, 7, 8, 9, 11, 14, 16]

    statTheta = np.arange(thetastart, thetaend, 0.01)
    statSigP = np.zeros((len(PJlist),len(statTheta)))
    statSigH = np.zeros((len(PJlist),len(statTheta)))
    statFAC = np.zeros((len(PJlist),len(statTheta)))
    statJx = np.zeros((len(PJlist),len(statTheta)))
    statVd = np.zeros((len(PJlist),len(statTheta)))
    statPele = np.zeros((len(PJlist),len(statTheta)))
    statPj = np.zeros((len(PJlist),len(statTheta)))

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(7.0, 9.0))
    ax[6].set_xlabel(r"$\Delta \theta$ (deg)")
    ax[0].set_xlim(thetastart, thetaend)
    ax[0].axhline(y=0, color='k', ls='--', lw=0.5)
    i=0
    for PJnum in PJlist:
        NC = 0
        if PJnum == 21:
            NC = 1
        tlim = utxings.Get_CrossingUTs(PJnum, NS, NC)
        name = ut.get_name(PJnum, NS, NC)
        if tlim[0] == 0 or tlim[0] == '0' :
            print("No crossing named", name + '.')
        else:
            print("Crossing", name, 'found !')
            dtstart = datetime.fromisoformat(tlim[0])
            dtend = datetime.fromisoformat(tlim[1])
            theta1, dtarr1, thetadB, dtarrdB, theta, dtarr, SIGH, SIGP, Jpara, Jxys, vDrift, TEFlux = get_data(PJnum, NS, NC)
            # then plots the data
            # ax[1]: Conductances
            ind1 = ((theta1 >= thetastart) & (theta1 <= thetaend) & (dtarr1 >= dtstart) & (dtarr1 <= dtend))
            ind2 = ((thetadB >= thetastart) & (thetadB <= thetaend) & (dtarrdB >= dtstart) & (dtarrdB <= dtend))
            ind5 = ((theta >= thetastart) & (theta <= thetaend) & (dtarr >= dtstart) & (dtarr <= dtend))
            lwidth = 1
            # When theta(time) is an increasing function in north
            # or a decreasing function in south 
            # regular situation for south, but non regular for north
            if (NS == 0)&(theta1[ind1][1] - theta1[ind1][0] > 0) or (NS == 1)&(theta1[ind1][1] - theta1[ind1][0] < 0):
                theta1_pl = theta1[ind1]
                SIGH_pl = SIGH[ind1]
                SIGP_pl = SIGP[ind1]
                theta_pl = theta[ind5]
                FAC_pl = Jpara[ind5]
                if NS == 1:
                ## /!\ Attention, traitement séparé de Nord et Sud !!!
                    print("careful ! ")
                    Jx_pl = - Jxys[0][ind5]
                else:
                ## /!\ Attention, fin de traitement séparé de Nord et Sud !!!
                    Jx_pl = Jxys[0][ind5]
                thetadB_pl = thetadB[ind2]
                Vd_pl = vDrift[ind2]
                Pele_pl = TEFlux[ind1]
                Pj_pl = Jxys[3][ind5]
            else:
                # When theta(time) is an decreasing function in north
                # or a increasing function in south
                # regular situation for north, but not for south
                theta1_pl = -np.flip(theta1[ind1])
                SIGH_pl = np.flip(SIGH[ind1])
                SIGP_pl = np.flip(SIGP[ind1])
                theta_pl = -np.flip(theta[ind5])
                FAC_pl = np.flip(Jpara[ind5])
                Pj_pl = np.flip(Jxys[3][ind5])
                if NS == 1:
                ## /!\ Attention, traitement séparé de Nord et Sud !!!
                    Jx_pl = -np.flip(Jxys[0][ind5])
                ## /!\ Attention, fin de traitement séparé de Nord et Sud !!!
                else:
                    print("careful ! ")
                    Jx_pl = np.flip(Jxys[0][ind5])
                
                thetadB_pl = -np.flip(thetadB[ind2])
                Vd_pl = np.flip(vDrift[ind2])
                Pele_pl = np.flip(TEFlux[ind1])
            ax[0].plot(theta1_pl, SIGH_pl, linewidth=lwidth)
            ax[1].plot(theta1_pl, SIGP_pl, linewidth=lwidth)
            ax[2].plot(theta_pl, FAC_pl, linewidth=lwidth)
            ax[3].plot(theta_pl, Jx_pl, linewidth=lwidth)
            ax[4].plot(thetadB_pl, Vd_pl, linewidth=lwidth)
            ax[5].plot(theta1_pl, Pele_pl, linewidth=lwidth, label=f'{PJnum}')
            ax[6].plot(theta_pl, Pj_pl, linewidth=lwidth)

            f_SigH = interp1d(theta1_pl, SIGH_pl, fill_value=(0,0), bounds_error=False)
            f_SigP = interp1d(theta1_pl, SIGP_pl, fill_value=(0,0), bounds_error=False)
            f_FAC = interp1d(theta_pl, FAC_pl, fill_value=(0,0), bounds_error=False)
            f_Jx = interp1d(theta_pl, Jx_pl, fill_value=(0,0), bounds_error=False)
            f_Vd = interp1d(thetadB_pl, Vd_pl, fill_value=(0,0), bounds_error=False)
            f_Pele = interp1d(theta1_pl, Pele_pl, fill_value=(0,0), bounds_error=False)
            f_Pj = interp1d(theta_pl, Pj_pl, fill_value=(0,0), bounds_error=False)

            if i!=20:
                statSigH[i] = f_SigH(statTheta)
                statSigP[i] = f_SigP(statTheta)
                statFAC[i] = f_FAC(statTheta)
                statJx[i] = f_Jx(statTheta)
                statVd[i] = f_Vd(statTheta)
                statPele[i] = f_Pele(statTheta)
                statPj[i] = f_Pj(statTheta)
            i+=1

    if NS == 0:
        ax[2].set_ylim(-1.5, 1.5)
        ax[3].set_ylim(-300, 700)
        ax[4].set_ylim(-2700, 1750)
        ax[0].set_ylim(-2, 35)
        ax[1].set_ylim(-0.5, 13)
        ax[5].set_ylim(-50, 200)
        ax[6].set_ylim(-50, 200)
    elif NS == 1:
        ax[2].set_ylim(-3, 3)
        ax[4].set_ylim(-11000, 10000)
        ax[0].set_ylim(-2, 25)
        ax[1].set_ylim(-0.5, 13)
        ax[5].set_ylim(-10, 1200)
        ax[6].set_ylim(-10, 1200)
    ax[0].set_ylabel(r'$\Sigma_{H, (\Omega^{-1})}$', labelpad=5)
    ax[1].set_ylabel(r'$\Sigma_{P, (\Omega^{-1})}$')
    ax[1].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[2].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[2].set_ylabel(r"$FAC_{(\mu A/m^2)}$")
    ax[3].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[3].set_ylabel(r'$J_{x, (mA/m)}$')
    ax[4].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[4].set_ylabel(r"$(E \times B)_{y, m/s}$")
    ax[5].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[5].set_ylabel(r'$P_{ele, (mW/m^2)}$')
    ax[6].axhline(y=0, color='k', ls='--', lw=0.5)
    ax[6].set_ylabel(r'$P_{J, (mW/m^2)}$')
    ax[0].annotate('(a)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[1].annotate('(b)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[2].annotate('(c)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[2].annotate('up', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.75))
    ax[2].annotate('down', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.2))
    ax[3].annotate('(d)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[3].annotate('equator', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.75))
    ax[3].annotate('pole', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.2))
    ax[4].annotate('(e)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[4].annotate('east', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.75))
    ax[4].annotate('west', xy=(0,0), xycoords='axes fraction', xytext=(0.92, 0.2))
    ax[5].annotate('(f)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[6].annotate('(g)', xy=(0,0), xycoords='axes fraction', xytext=(0.05, 0.75))
    ax[6].annotate('pole', xy=(0,0), xycoords='axes fraction', xytext=(0, -0.35))
    ax[6].annotate('equator', xy=(0,0), xycoords='axes fraction', xytext=(0.92, -0.35))


    # Statistics
    stats = np.zeros((3, 7, len(statTheta)))
    # if NS == 0:
    stats[0][0] = np.nanmedian(statSigH[1:], axis = 0)
    stats[0][1] = np.nanmedian(statSigP[1:], axis = 0)
    stats[0][2] = np.nanmedian(statFAC[1:], axis = 0)
    stats[0][3] = np.nanmedian(statJx[1:], axis = 0)
    stats[0][4] = np.nanmedian(statVd[1:], axis = 0)
    stats[0][5] = np.nanmedian(statPele[1:], axis = 0)
    stats[0][6] = np.nanmedian(statPj[1:], axis = 0)
    stats[1][0] = np.nanquantile(statSigH[1:], 0.8, axis = 0)
    stats[1][1] = np.nanquantile(statSigP[1:], 0.8, axis = 0)
    stats[1][2] = np.nanquantile(statFAC[1:], 0.8, axis = 0)
    stats[1][3] = np.nanquantile(statJx[1:], 0.8, axis = 0)
    stats[1][4] = np.nanquantile(statVd[1:], 0.8, axis = 0)
    stats[1][5] = np.nanquantile(statPele[1:], 0.8, axis = 0)
    stats[1][6] = np.nanquantile(statPj[1:], 0.8, axis = 0)
    stats[2][0] = np.nanquantile(statSigH[1:], 0.2, axis = 0)
    stats[2][1] = np.nanquantile(statSigP[1:], 0.2, axis = 0)
    stats[2][2] = np.nanquantile(statFAC[1:], 0.2, axis = 0)
    stats[2][3] = np.nanquantile(statJx[1:], 0.2, axis = 0)
    stats[2][4] = np.nanquantile(statVd[1:], 0.2, axis = 0)
    stats[2][5] = np.nanquantile(statPele[1:], 0.2, axis = 0)
    stats[2][6] = np.nanquantile(statPj[1:], 0.2, axis = 0)
    for i in range(7):
        if NS == 1:
            ax[i].plot(statTheta, stats[0][i], 'k', linewidth=1.5)
            ax[i].plot(statTheta, stats[1][i], 'k:', linewidth=0.8)
            ax[i].plot(statTheta, stats[2][i], 'k:', linewidth=0.8)
            ax[i].fill_between(statTheta, stats[1][i], stats[2][i],  alpha=0.2, color='k') # if median
        else: 
            if i not in [2, 3, 4]:
                ax[i].plot(statTheta, stats[0][i], 'k', linewidth=1.5)
                ax[i].plot(statTheta, stats[1][i], 'k:', linewidth=0.8)
                ax[i].plot(statTheta, stats[2][i], 'k:', linewidth=0.8)
                ax[i].fill_between(statTheta, stats[1][i], stats[2][i],  alpha=0.2, color='k') # if median

    for i in range(7):
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    ax[6].xaxis.set_minor_locator(AutoMinorLocator())

    fig.align_ylabels()
    fig.legend(loc='upper right')
    plt.tight_layout()
    if not show:
        if NS == 0:
            fig.suptitle("Key parameters for Northern crossings")
        elif NS == 1:
            fig.suptitle("Key parameters for Southern crossings")
        if NS == 0:
            plt.savefig(hp.pathtoplots + 'superposed/DeltaThetaNorth_med.png')
        elif NS == 1:
            plt.savefig(hp.pathtoplots + 'superposed/DeltaThetaSouth_med.png')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__' and True:
    show=False
    # plotSuperposed(0, show=show)
    plotSuperposed(1, show=show)
