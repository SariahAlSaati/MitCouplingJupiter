import hyperparameters as hp
import UTLutils as ut
import UTLcrossings as utxings
import UTLmag as utmag
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os.path
import requests
from scipy import signal
from scipy.io import readsav


mu0 = 4*np.pi*1.0e-7


def DownloadMag(PJnum, NS, NC):
    success = False
    datelist = utxings.Get_MainOvalCrossUTs(
        PJnum, NS, NC, OffSet_min=hp.Offset_min)
    datetimestart = datetime.fromisoformat(datelist[0])

    if PJnum in [7, 15, 21, 22, 29, 30]:
        dayOfYear = (datetimestart - timedelta(days=1)).strftime('%Y%j')
    else:
        dayOfYear = datetimestart.strftime('%Y%j')

    filen = 'mag/fgm_jno_l3_' + dayOfYear + 'pc_pj' + \
        str(PJnum).zfill(2) + '_r1s_v02.sts'

    url = 'https://pds-ppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/JNO-J-3-FGM-CAL-V1.0/DATA/JUPITER/PC/PERI-'
    url += str(PJnum).zfill(2) + '/' + filen
    print("Download initiated, please wait until file downloaded.")
    r = requests.get(url, allow_redirects=True)
    print("File downloaded.")
    open(hp.pathtoinstruments + filen, 'wb').write(r.content)
    print("File saved.")
    success = True
    return success


def mag(PJnum, NS, NC):
    # data written as:
    # year, doy, hour, min, s, ms, decimal_doy, Bx, By, Bz, Brange, x, y, z
    # in planetocentric coordinates
    # unites in nanoTesla and kiloMeters
    # data written in SIIIRH coordinates

    alpha = utxings.get_alpha(PJnum, NS, NC)

    datelist = utxings.Get_MainOvalCrossUTs(
        PJnum, NS, NC, OffSet_min=hp.Offset_min)
    dtstart = datetime.fromisoformat(datelist[0])
    dtend = datetime.fromisoformat(datelist[1])

    if PJnum in [7, 15, 21, 22, 29, 30]:
        dayOfYear = (dtstart - timedelta(days=1)).strftime('%Y%j')
    else:
        dayOfYear = dtstart.strftime('%Y%j')

    filen = 'mag/fgm_jno_l3_' + dayOfYear + \
        'pc_pj' + str(PJnum).zfill(2) + '_r1s_v02.sts'

    if not os.path.isfile(hp.pathtoinstruments + filen):
        print("File does not exist, trying to download it.")
        success = DownloadMag(PJnum, NS, NC)
        if not success:
            print("Did not manage to download file. Please do it manually.")
            return 0

    magdata = np.transpose(np.loadtxt(
        hp.pathtoinstruments + filen, skiprows=126))

    dtarr = []
    n = len(magdata[0])
    for i in range(n):
        strdate = str(int(magdata[0][i])).zfill(4) + str(int(magdata[1][i])).zfill(3) + str(int(magdata[2][i])).zfill(
            2) + str(int(magdata[3][i])).zfill(2) + str(int(magdata[4][i])).zfill(2) + str(int(magdata[5][i])).zfill(3)
        dt = datetime.strptime(strdate, '%Y%j%H%M%S%f')
        dtarr.append(dt)
    dtarr = np.array(dtarr)

    ddoy = magdata[6]
    Bx = magdata[7]
    By = magdata[8]
    Bz = magdata[9]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    x = magdata[11]/hp.Rj
    y = magdata[12]/hp.Rj
    z = magdata[13]/hp.Rj
    posi1 = np.transpose(np.array([x, y, z]))
    sph1 = np.transpose(ut.Car2SphV(posi1, Bx, By, Bz))
    Br, Bt, Bp = sph1[0], sph1[1], sph1[2]
    sph = np.transpose(ut.Car2SphP(x, y, z))
    r, theta, phi = sph[0], sph[1], sph[2]  # degrees
    Br2, Bt2, Bp2, Bmag2 = utmag.GetJupiterMag(r, theta, phi, CAN=1)

    dBr = Br - Br2
    dBt = Bt - Bt2
    dBp = Bp - Bp2

    # selecting the 30 seconds window and apply moving averaging
    tlim = utxings.Get_CrossingUTs(PJnum, NS, NC, OffSet_min=hp.Offset_min)
    dtsobs = datetime.fromisoformat(tlim[0])
    dtsend = datetime.fromisoformat(tlim[1])
    w = 30
    dBr_avg = ut.moving_average(dBr, w)
    dBt_avg = ut.moving_average(dBt, w)
    dBp_avg = ut.moving_average(dBp, w)
    fcut = 1/((dtsend - dtsobs).total_seconds() * alpha)

    freq = 1/((dtarr[1] - dtarr[0]).total_seconds())
    b, a = signal.butter(4, fcut, 'hp', fs=freq)
    dBr_leclerc = signal.filtfilt(b, a, dBr_avg)
    dBt_leclerc = signal.filtfilt(b, a, dBt_avg)
    dBp_leclerc = signal.filtfilt(b, a, dBp_avg)

    indexofinterest = ((dtarr > dtstart) & (dtarr < dtend))
    dtarr = dtarr[indexofinterest]
    Br = Br[indexofinterest]
    Br2 = Br2[indexofinterest]
    Bp = Bp[indexofinterest]
    Bp2 = Bp2[indexofinterest]
    Bt = Bt[indexofinterest]
    Bt2 = Bt2[indexofinterest]
    dBr = dBr[indexofinterest]
    dBt = dBt[indexofinterest]
    dBp = dBp[indexofinterest]
    dBr_avg = dBr_avg[indexofinterest]
    dBt_avg = dBt_avg[indexofinterest]
    dBp_avg = dBp_avg[indexofinterest]
    Bz = Bz[indexofinterest]
    Bmag = Bmag[indexofinterest]
    dBr_leclerc = dBr_leclerc[indexofinterest]
    dBt_leclerc = dBt_leclerc[indexofinterest]
    dBp_leclerc = dBp_leclerc[indexofinterest]

    name = ut.get_name(PJnum, NS, NC)

    Juls_Bp = ut.datetime2julian(dtarr)
    filename = "dB_" + name
    np.savez(hp.pathtoresB + filename, Juls_Bp=Juls_Bp,
             dBr_nT=dBr_leclerc, dBt_nT=dBt_leclerc, dBp_nT=dBp_leclerc, Br=Br, Bmag=Bmag)
    return dtarr, dBr_leclerc, dBt_leclerc, dBp_leclerc, Br, Bmag


def eflux(PJnum, NS, NC):
    name = ut.get_name(PJnum, NS, NC)
    # get jade data
    filenamejade = hp.pathtoinstruments + 'jade/JADE_' + name + '.asc'

    data_jade = pd.read_csv(filenamejade, header=None, skiprows=80)
    e0_kev_jade = data_jade[2].values * 1e-3
    ne = 0
    while e0_kev_jade[ne+1] > e0_kev_jade[ne]:
        ne += 1
    ne += 1
    e0_kev_jade = e0_kev_jade[:ne]

    time_jade = np.reshape(data_jade[0].values, (-1, ne))[:, 0]
    nt = len(time_jade)
    for i in range(nt):
        time_jade[i] = datetime.fromisoformat(time_jade[i][:-1])

    # get jedi data
    filenamejedi = hp.pathtoinstruments + 'jedi/JEDI_' + name + '.d2s'

    data_jedi = pd.read_csv(filenamejedi, skiprows=9,
                            delim_whitespace=True, header=None)

    time_jedi = data_jedi[0].values
    for i in range(len(time_jedi)):
        time_jedi[i] = datetime.fromisoformat(time_jedi[i][4:])

    if PJnum >= 20:
        e0_kev_jedi = np.array([14.464938575, 18.126812285, 19.20733568, 20.795694495, 22.26813372, 24.299367314999998, 28.065854545, 33.21929375, 39.379541059999994, 47.051023485, 56.37939779, 67.66830743, 81.26822621, 97.497416805, 117.39113505,
                               132.0921508, 144.8166754, 158.41077085, 174.16107005, 190.8836813, 209.44072484999998, 229.00415034999997, 250.84985275, 292.11848335, 348.92191225, 416.93224814999996, 499.19747135, 599.91776015, 725.57689435, 1001.1622981])
    else:
        e0_kev_jedi = np.array([14.305, 18.945, 22.61, 27.055, 32.07, 37.67, 44.82, 53.9, 64.905, 78.18, 94.04499999999999, 113.52,
                               136.81, 164.76999999999998, 198.5, 238.255, 285.27, 340.895, 407.13, 487.005, 584.575, 705.9300000000001, 1002.485])

    ne0 = len(e0_kev_jedi)
    pflux_jade = np.transpose(np.reshape(
        data_jade[5].values, (-1, ne)))/1000*np.pi
    pflux_jedi = np.transpose(
        data_jedi[[i for i in range(1, ne0 + 1)]].to_numpy())/1000*np.pi
    # pflux is c/s/sr/cm2/keV
    # returns data in c/s/cm2/eV

    # fuse
    # first step: get them under same time
    eps = timedelta(seconds=0.5)
    intersectionjade = (
        (time_jade >= time_jedi[0]-eps) & (time_jade <= time_jedi[-1]+eps))
    intersectionjedi = (
        (time_jedi >= time_jade[0]-eps) & (time_jedi <= time_jade[-1]+eps))
    time_jade = time_jade[intersectionjade]
    time_jedi = time_jedi[intersectionjedi]
    pflux_jade = np.transpose(np.transpose(pflux_jade)[intersectionjade])
    pflux_jedi = np.transpose(np.transpose(pflux_jedi)[intersectionjedi])
    time_common = time_jedi
    e0_kev = np.append(e0_kev_jade, e0_kev_jedi)
    pflux = np.block([[pflux_jade], [pflux_jedi]])
    indjade = e0_kev_jade < 30
    indjedi = 30 < e0_kev_jedi
    e0_kev_jade = e0_kev_jade[indjade]
    e0_kev_jedi = e0_kev_jedi[indjedi]
    pflux_jade = pflux_jade[indjade]
    pflux_jedi = pflux_jedi[indjedi]
    e0_kev = np.append(e0_kev_jade, e0_kev_jedi)
    pflux = np.block([[pflux_jade], [pflux_jedi]])
    time_common = np.copy(time_jedi)

    return time_common, e0_kev, pflux, pflux_jade, pflux_jedi


def get_UVS_profile(PJ, NS, NC):
    if PJ <= 22:
        path = hp.pathtoinstruments + 'uvs/Juno_brightness_profiles_highres/'
        name = f"brillance_profil_50_1000_jrm09_PJ{PJ}_test_new_interp_66per_5.idl"

        file = readsav(path + name)
        brightness = file['bri_h2ly_jrm09'][0]
        brightness = np.array(brightness)
        brightness[brightness < -995] = np.nan
        timeUVS = file['time_ephem_out'][file['a2'].astype(int)]
        tlim = utxings.Get_CrossingUTs(PJ, NS, NC)

        dayiso = tlim[0][:10]
        dayUVS = datetime.strptime(dayiso, "%Y-%m-%d")
        time = []
        if timeUVS[-1] > 24:
            dayUVS = dayUVS - timedelta(days=1)
        for i in range(len(timeUVS)):
            td = timeUVS[i].item()
            time.append(dayUVS + timedelta(hours=td))
    else:  # no data
        tlim = utxings.Get_CrossingUTs(PJ, NS, NC)
        dayUVS = datetime.strptime(tlim[0], "%Y-%m-%dT%H:%M:%S.000")
        time = [dayUVS - timedelta(seconds=i) for i in range(5)]
        brightness = np.zeros(len(time))
    return np.array(time), brightness*8.1  # To get total H2 brightness in kR


if __name__ == "__main__":
    pass
