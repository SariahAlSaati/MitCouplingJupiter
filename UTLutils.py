import hyperparameters as hp
import spiceypy
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d


def get_name(pj, ns, nc):
    return "PJ" + str(pj).zfill(2) + hp.NS[ns] + str(nc)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def get_crossing(name):
    pj = int(name[2:4])
    if name[4] == 'N':
        ns = 0
    else:
        ns = 1
    nc = int(name[5])
    return pj, ns, nc


@np.vectorize
def doy_to_ymd(doy, hours, minutes, second):
    return datetime.strptime(doy+hours+minutes+second, '%Y%j%H%M%S')


@np.vectorize
def doy_float_to_ymd(doy, hours):
    ymd = datetime.strptime((doy)[0:7], '%Y%j')
    daydec = timedelta(hours=(hours))
    return (ymd + daydec)


@np.vectorize
def return_hour_minute(date):
    # Author: Corentin Louis, October 2021
    result = datetime.strftime(date, '%H:%M')
    return (result)


def julday(month, day, year, hour=0, minute=0, second=0):
    # From pytesmo.timedate.julian by Sariah Al Saati on 15-04-2021.
    # Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
    # All rights reserved.
    """
    Julian date from month, day, and year (can be scalars or arrays)

    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    hour : numpy.ndarray or int32, optional
        Hour.
    minute : numpy.ndarray or int32, optional
        Minute.
    second : numpy.ndarray or int32, optional
        Second.

    Returns
    -------
    jul : numpy.ndarray or double
        Julian day.
    """
    month = np.array(month)
    day = np.array(day)
    inJanFeb = month <= 2
    jy = year - inJanFeb
    jm = month + 1 + inJanFeb * 12

    jul = np.int32(np.floor(365.25 * jy) +
                   np.floor(30.6001 * jm) + (day + 1720995.0))
    ja = np.int32(0.01 * jy)
    jul += 2 - ja + np.int32(0.25 * ja)

    jul = jul + hour / 24.0 - 0.5 + minute / 1440.0 + second / 86400.0

    return jul


def julian2date(julian):
    # From pytesmo.timedate.julian by Sariah Al Saati on 15-04-2021.
    # Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
    # All rights reserved.
    """
    Calendar date from julian date.
    Works only for years past 1582!

    Parameters
    ----------
    julian : numpy.ndarray or double
        Julian day.

    Returns
    -------
    year : numpy.ndarray or int32
        Year.
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    hour : numpy.ndarray or int32
        Hour.
    minute : numpy.ndarray or int32
        Minute.
    second : numpy.ndarray or int32
        Second.
    """
    min_julian = 2299160
    max_julian = 1827933925

    julian = np.atleast_1d(np.array(julian, dtype=float))

    if np.min(julian) < min_julian or np.max(julian) > max_julian:
        raise ValueError("Value of Julian date is out of allowed range.")

    jn = (np.round(julian + 0.0000001)).astype(np.int32)

    jalpha = (((jn - 1867216) - 0.25) / 36524.25).astype(np.int32)
    ja = jn + 1 + jalpha - (np.int32(0.25 * jalpha))
    jb = ja + 1524
    jc = (6680.0 + ((jb - 2439870.0) - 122.1) / 365.25).astype(np.int32)
    jd = (365.0 * jc + (0.25 * jc)).astype(np.int32)
    je = ((jb - jd) / 30.6001).astype(np.int32)

    day = jb - jd - np.int64(30.6001 * je)
    month = je - 1
    month = (month - 1) % 12 + 1
    year = jc - 4715
    year = year - (month > 2)

    fraction = (julian + 0.5 - jn).astype(np.float64)
    eps = (np.float64(1e-12) * np.abs(jn)).astype(np.float64)
    eps.clip(min=np.float64(1e-12), max=None)
    hour = (fraction * 24. + eps).astype(np.int64)
    hour.clip(min=0, max=23)
    fraction -= hour / 24.
    minute = (fraction * 1440. + eps).astype(np.int64)
    minute = minute.clip(min=0, max=59)
    second = (fraction - minute / 1440.) * 86400.
    second = second.clip(min=0, max=None)
    microsecond = ((second - np.int32(second)) * 1e6).astype(np.int32)
    microsecond = microsecond.clip(min=0, max=999999)
    second = second.astype(np.int32)

    return year, month, day, hour, minute, second, microsecond


def julian2datetime(julian):
    year, month, day, hour, minute, second, microsecond = julian2date(julian)
    res = []
    for i in range(len(year)):
        res.append(datetime(year[i], month[i], day[i],
                   hour[i], minute[i], second[i], microsecond[i]))
    return np.array(res)


def datetime2julian(dtarr):
    n = len(dtarr)
    day = np.array([dtarr[i].day for i in range(n)])
    month = np.array([dtarr[i].month for i in range(n)])
    year = np.array([dtarr[i].year for i in range(n)])
    hour = np.array([dtarr[i].hour for i in range(n)])
    minute = np.array([dtarr[i].minute for i in range(n)])
    second = np.array([dtarr[i].second for i in range(n)])
    return julday(month, day, year, hour, minute, second)


def datetimeToThetaFP(dtarr, thetaFP, dt2):
    jul = datetime2julian(dtarr)
    jul2 = datetime2julian(dt2)
    f = interp1d(jul, thetaFP, fill_value="extrapolate")
    thetaFP2 = f(jul2)
    return thetaFP2


def datetime2utc(dt):
    jul = datetime2julian(dt)
    utc = np.array([yxjul2utc(j) for j in jul])
    return utc


def deg2rad(alpha):
    return alpha*np.pi/180


def rad2deg(alpha):
    return alpha * 180/np.pi


def yxdiff(x):
    # This pro is to get the diff(x), ie, diff([1,2,3,1]) = [1, 1, -2]
    n = len(x)
    res = []
    for i in range(n-1):
        res.append(x[i+1] - x[i])
    return res


def yxjul2utc(jul):
    # transfer from Julday, to utc
    # [1] Written by Yuxian Wang 2020-01-11 00:18:33
    Julday = float(jul)   # make sure in double

    Year, Month, Day, Hour, Minute, Second, NanoSec = julian2date(Julday)
    MiliSec = int(NanoSec*1e-3)

    UTC = str(Year[0]).zfill(4) + '-'
    UTC += str(Month[0]).zfill(2) + '-'
    UTC += str(Day[0]).zfill(2) + '/'
    UTC += str(Hour[0]).zfill(2) + ':'
    UTC += str(Minute[0]).zfill(2) + ':'
    UTC += str(Second[0]).zfill(2) + '.'
    UTC += str(MiliSec).zfill(3)

    # 2021-04-09/10:26:33.000
    return UTC


def yxutc2jul(utc):

    # This function is to transform from utc date to Julday.
    # Written by Yuxian Wang 2020-01-11 00:06:41

    Year = int(utc[:4])
    Month = int(utc[5:7])
    Day = int(utc[8:10])
    Hour = int(utc[11:13])
    Minute = int(utc[14:16])
    Second = int(utc[17:19])
    Julday = julday(Month, Day, Year, Hour, Minute, Second)

    return Julday


def Car2SphP(xp, yp, zp):
    # ==========================================================
    #  Car2SphP.pro
    # ==========================================================
    # Cartesian to Spherical coordinates for positions.
    # Note: there are differences between points transformation and that of vectors.
    # Function:
    #   input: x, y, z (arrays) in cartesian coords.
    #   output: r, theta, phi in degrees
    x = np.array(xp)
    y = np.array(yp)
    z = np.array(zp)
    n = len(x)
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-15*np.ones(n))

    theta = np.arccos(z/r)*180/np.pi
    phi = np.arctan2(y, x)*180/np.pi

    # transformation is safe with np.arctan2

    # [correction for phi not necessary with numpy.arctan2, omitting this part of the code from original code]
    # ir = np.where(x < 0)[0]
    # if len(ir) > 0 :
    #     phi[ir] += 180.0

    ir = np.where(phi < 0)[0]
    if len(ir) > 0:
        phi[ir] += 360.0

    return np.transpose(np.array([r, theta, phi]))


def Car2SphV(posi, Bx, By, Bz):
    # ==========================================================
    #  Car2SphV.pro
    # ==========================================================
    # Cartesian to Spherical coordinates for vectors
    # Note: there are differences between points transformation and that of vectors.
    # Function:
    #   input:  x, y, z in cartesian coords., Bx, By, Bz
    #   output: Br, Bt, Bp
    #

    posi = np.array(posi)
    x = posi[:, 0]
    y = posi[:, 1]
    z = posi[:, 2]

    Bx = np.array(Bx)
    By = np.array(By)
    Bz = np.array(Bz)

    tmp = Car2SphP(x, y, z)
    theta = tmp[:, 1]*np.pi/180
    phi = tmp[:, 2]*np.pi/180

    sint = np.sin(theta)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    cosp = np.cos(phi)

    Br = Bx*sint*cosp + By*sint*sinp + Bz*cost
    Bt = Bx*cost*cosp + By*cost*sinp - Bz*sint
    Bp = -Bx*sinp + By*cosp

    return np.transpose(np.array([Br, Bt, Bp]))


def Sph2CarP(rp, thetap, phip):
    r = np.array(rp)
    theta = np.array(thetap)*np.pi/180
    phi = np.array(phip)*np.pi/180

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return np.transpose(np.array([x, y, z]))


def Sph2CarV(posi, Br, Bt, Bp):
    posi = np.array(posi)
    theta = posi[:, 1]*np.pi/180
    phi = posi[:, 2]*np.pi/180

    sint = np.sin(theta)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    cosp = np.cos(phi)

    Bx = Br*sint*cosp + Bt*cost*cosp - Bp*sinp
    By = Br*sint*sinp + Bt*cost*sinp + Bp*cosp
    Bz = Br*cost - Bt*sint

    return np.transpose([Bx, By, Bz])


def Get_JunoState(utc_in, coords):
    # using spicepy
    # Annex et al., (2020). SpiceyPy: a Pythonic Wrapper for the SPICE Toolkit.
    # Journal of Open Source Software, 5(46), 2050, https://doi.org/10.21105/joss.02050
    # https://github.com/AndrewAnnex/SpiceyPy

    # Adapted from the code written by Yuxian Wang in IDL
    # All rights reserved
    # ============================

    # This function is to get the Juno state infor. including position in 'jso' or 'iau' coords.
    #  for given dates. To successfully run this pro, SPICE kernel is needed for IDL.
    # Validate Example:
    #       utc = yxdoy2ymd(214.238211756, 2016)
    #       FGM: 38268.292   -8066824.031    -767763.602
    #       Get_JunoState(utc_in, 'jso')
    # Written by Yuxian Wang 2020-05-06 00:15:38
    # ============================

    # [1] Load a set of kernels: an SPK file, a PCK file and a leapseconds file
    spiceypy.spiceypy.furnsh(hp.pathtodata + 'juno_posi.tm')
    RJ = 71492.0

    # [2] Time Transfer: UTC -> Ephemeris time
    utc = utc_in
    TDB = spiceypy.spiceypy.str2et(utc)

    # [3] Calculate the state and time [x_km, y, z, vx_km_s, vy, vz]
    if coords == 'jso':
        STATE, ltime = spiceypy.spiceypy.spkezr(
            'Juno', TDB, 'JUNO_JSO', 'LT+S', 'Jupiter')
        jso_xyz_out = np.array(STATE)[:, 0:3] / RJ
        res = jso_xyz_out
    elif coords == 'jss':
        STATE, ltime = spiceypy.spiceypy.spkezr(
            'Juno', TDB, 'JUNO_JSS', 'LT+S', 'Jupiter')
        jss_xyz_out = np.array(STATE)[:, 0:3] / RJ
        res = jss_xyz_out
    elif coords == 'iau':
        STATE, ltime = spiceypy.spiceypy.spkezr(
            'Juno', TDB, 'IAU_JUPITER', 'LT+S', 'Jupiter')
        iau_xyz_out = np.array(STATE)[:, 0:3] / RJ
        res = iau_xyz_out
    else:
        raise ValueError(
            "Please select valid value for coords in ['jso', 'jss', 'iau']")

    spiceypy.spiceypy.kclear()   # unload all kernels
    return res


def Get_DriftVelocity(julsB, Br, Bmag, julsE, Ex):
    """ Calculate drift velocity in the y axis using Bz data in nT 
    and Ex data in mV/m.  data is returned with magnetic datetime."""
    f_Ex = interp1d(julsE, Ex*1e-3, fill_value="extrapolate")
    Ex_new = f_Ex(julsB)
    vDrift = -(Ex_new*Br*1e-9)/(Bmag*1e-9)**2
    return vDrift


def Get_DriftVelocity2(julsB, Br, Bmag, julsE, Ex):
    """ Calculate drift velocity in the y axis using Bz data in nT 
    and Ex data in mV/m.  data is returned with magnetic datetime."""
    f_Bmag = interp1d(julsB, Bmag, fill_value="extrapolate")
    Bmag_new = f_Bmag(julsE)
    f_Br = interp1d(julsB, Br, fill_value="extrapolate")
    Br_new = f_Br(julsE)
    vDrift = -(Ex*1e-3*Br_new*1e-9)/(Bmag_new*1e-9)**2
    return vDrift


if __name__ == '__main__':
    pass
