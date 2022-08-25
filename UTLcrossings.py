import UTLutils as ut
import numpy as np
from datetime import datetime


def Get_MainOvalCrossUTs(PJnum, NS, NC, OffSet_min=None):
    # +
    # =================================================
    # Description: Get the time ranges of the main oval crossing for North-0 and South-1.
    #    The main oval crossings periods for the first 9 perijovs, partly based on the table 1
    #    of Allegrini et al. (2020).
    # Input:
    #    Set OffSet_min to extend the time range.
    # Output:
    # Example:
    # History:
    # =================================================
    # -
    if NC == 3:
        if PJnum == 5:
            return ['2017-03-27T09:28:00.000', '2017-03-27T09:33:00.000']
        if PJnum == 13:
            return ['2018-05-24T05:11:00.000', '2018-05-24T05:18:00.000']
        if PJnum == 22:
            return ['2019-09-12T03:16:00.000', '2019-09-12T03:22:00.000']
        else:
            return [0, 0]

    if PJnum == 1:
        tlim = np.array([
            ['2016-08-27T11:50:00.000', '2016-08-27T12:20:00.000'],
            [0, 0],
            ['2016-08-27T13:20:00.000', '2016-08-27/13:45:00.000'],
            [0, 0]])
    elif PJnum == 3:
        tlim = np.array([
            ['2016-12-11T16:10:00.000', '2016-12-11T16:30:00.000'],
            [0, 0],
            ['2016-12-11T17:30:00.000', '2016-12-11T17:45:00.000'],
            [0, 0]])
    elif PJnum == 4:
        tlim = np.array([
            ['2017-02-02T12:15:00.000', '2017-02-02T12:30:00.000'],
            [0, 0],
            ['2017-02-02T13:30:00.000', '2017-02-02T13:45:00.000'],
            [0, 0]])
    elif PJnum == 5:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2017-03-27T09:27:00.000', '2017-03-27T09:55:00.000'],
            [0, 0]])
    elif PJnum == 6:
        tlim = np.array([
            [0, 0],
            [0, 0],
            # previously ['2017-05-19/06:30:00.000', '2017-05-19/07:30:00.000'],
            ['2017-05-19/06:48:00.000', '2017-05-19/07:05:00.000'],
            [0, 0]])
    elif PJnum == 7:
        tlim = np.array([
            ['2017-07-11T01:10:00.000', '2017-07-11T01:25:00.000'],
            [0, 0],
            ['2017-07-11T02:25:00.000', '2017-07-11T02:45:00.000'],
            [0, 0]])
    elif PJnum == 8:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2017-09-01T22:15:00.000', '2017-09-01/22:33:00.000'],
            [0, 0]])
    elif PJnum == 9:
        tlim = np.array([
            ['2017-10-24T17:14:00.000', '2017-10-24T17:25:00.000'],
            [0, 0],
            ['2017-10-24T18:22:00.000', '2017-10-24/18:45:00.000'],
            [0, 0]])
    elif PJnum == 11:
        tlim = np.array([
            ['2018-02-07T12:50:00.000', '2018-02-07T13:10:00.000'],
            [0, 0],
            ['2018-02-07T14:40:00.000', '2018-02-07T14:55:00.000'],
            [0, 0]])
    elif PJnum == 12:
        tlim = np.array([
            ['2018-04-01T09:00:00.000', '2018-04-01T09:15:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])
    elif PJnum == 13:
        tlim = np.array([
            # previously ['2018-05-24T05:00:00.000', '2018-05-24T05:20:00.000'],
            ['2018-05-24T05:00:00.000', '2018-05-24T05:15:00.000'],
            [0, 0],
            ['2018-05-24T06:15:00.000', '2018-05-24T06:30:00.000'],
            [0, 0]])
    elif PJnum == 14:
        tlim = np.array([
            ['2018-07-16T04:35:00.000', '2018-07-16T04:50:00.000'],
            [0, 0],
            ['2018-07-16T06:00:00.000', '2018-07-16T06:45:00.000'],
            [0, 0]])
    elif PJnum == 16:
        tlim = np.array([
            ['2018-10-29T20:05:00.000', '2018-10-29T20:25:00.000'],
            [0, 0],
            ['2018-10-29T21:50:00.000', '2018-10-29T22:05:00.000'],
            [0, 0]])
    elif PJnum == 18:
        tlim = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])
    elif PJnum == 20:
        tlim = np.array([
            ['2019-05-29T06:10:00.000', '2019-05-29T07:00:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])
    elif PJnum == 21:
        tlim = np.array([
            ['2019-07-21T02:55:00.000', '2019-07-21T03:18:00.000'],
            ['2019-07-21T03:35:00.000', '2019-07-21T03:45:00.000'],
            [0, 0],
            [0, 0]])
    elif PJnum == 22:
        tlim = np.array([
            ['2019-09-12T03:10:00.000', '2019-09-12T03:25:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])
    elif PJnum == 24:
        tlim = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])
    elif PJnum == 26:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2020-04-10T15:00:00.000', '2020-04-10T15:15:00.000'],
            [0, 0]])
    elif PJnum == 28:
        tlim = np.array([
            ['2020-07-25T05:30:00.000', '2020-07-25T05:40:00.000'],
            ['2020-07-25T05:55:00.000', '2020-07-25T06:00:00.000'],
            ['2020-07-25T07:10:00.000', '2020-07-25T07:40:00.000'],
            [0, 0]])
    elif PJnum == 29:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2020-09-16T03:15:00.000', '2020-09-16T03:45:00.000'],
            [0, 0]])
    elif PJnum == 30:
        tlim = np.array([
            ['2020-11-08T01:10:00.000', '2020-11-08T01:30:00.000'],
            [0, 0],
            ['2020-11-08T02:45:00.000', '2020-11-08T03:10:00.000'],
            [0, 0]])
    else:
        tlim = np.zeros((4, 2))

    # gets North crossing (first row, NS = 0) or South crossing (second row, NS = 1)
    tlim = tlim[2*NS + NC]

    if OffSet_min:
        OffSet_day = np.array(OffSet_min).astype(np.float) / 60.0/24.0
        jultlim = [ut.yxutc2jul(tlim[0]), ut.yxutc2jul(tlim[1])]
        return ut.yxjul2utc(jultlim[0] - OffSet_day[0]), ut.yxjul2utc(jultlim[1] + OffSet_day[1])

    return tlim


def Get_CrossingUTs(PJnum, NS, NC, OffSet_min=None):
    # fonction used for plotting purposes, can be modified as seen fit
    # This will not change the results of the analysis
    tlim = Get_MainOvalCrossUTs(PJnum, NS, NC, OffSet_min=OffSet_min)
    if NC == 3:
        if PJnum == 5:
            return ['2017-03-27T09:28:00.000', '2017-03-27T09:33:00.000']
        if PJnum == 13:
            return ['2018-05-24T05:11:00.000', '2018-05-24T05:15:00.000']
        if PJnum == 22:
            return ['2019-09-12T03:16:00.000', '2019-09-12T03:22:00.000']
        else:
            return [0, 0]
    if PJnum == 1:
        tlim = np.array([
            ['2016-08-27T12:05:00.000', '2016-08-27T12:20:00.000'],
            [0, 0],
            ['2016-08-27T13:20:00.000', '2016-08-27/13:45:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 3:
        tlim = np.array([
            ['2016-12-11T16:20:00.000', '2016-12-11T16:30:00.000'],
            [0, 0],
            ['2016-12-11T17:32:30.000', '2016-12-11T17:42:30.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 4:
        tlim = np.array([
            ['2017-02-02T12:20:00.000', '2017-02-02T12:30:00.000'],
            [0, 0],
            ['2017-02-02T13:30:00.000', '2017-02-02T13:45:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 5:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2017-03-27T09:35:00.000', '2017-03-27T09:55:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 6:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2017-05-19/06:48:30.000', '2017-05-19/07:05:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 7:
        tlim = np.array([
            ['2017-07-11T01:10:00.000', '2017-07-11T01:24:00.000'],
            [0, 0],
            ['2017-07-11T02:25:00.000', '2017-07-11T02:45:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 8:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2017-09-01T22:20:00.000', '2017-09-01/22:32:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 9:
        tlim = np.array([
            ['2017-10-24T17:15:00.000', '2017-10-24T17:23:00.000'],
            [0, 0],
            ['2017-10-24T18:23:00.000', '2017-10-24/18:45:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 11:
        tlim = np.array([
            ['2018-02-07T12:54:00.000', '2018-02-07T13:05:00.000'],
            [0, 0],
            ['2018-02-07T14:40:00.000', '2018-02-07T14:55:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 12:
        tlim = np.array([
            ['2018-04-01T09:02:30.000', '2018-04-01T09:10:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])[2*NS + NC]
    elif PJnum == 13:
        tlim = np.array([
            ['2018-05-24T05:00:00.000', '2018-05-24T05:15:00.000'],
            [0, 0],
            ['2018-05-24T06:15:00.000', '2018-05-24T06:30:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 14:
        tlim = np.array([
            ['2018-07-16T04:35:00.000', '2018-07-16T04:45:00.000'],
            [0, 0],
            ['2018-07-16T06:05:00.000', '2018-07-16T06:20:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 16:
        tlim = np.array([
            ['2018-10-29T20:05:00.000', '2018-10-29T20:19:00.000'],
            [0, 0],
            ['2018-10-29T21:50:00.000', '2018-10-29T22:05:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 20:
        tlim = np.array([
            ['2019-05-29T06:20:00.000', '2019-05-29T06:50:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])[2*NS + NC]
    elif PJnum == 21:
        tlim = np.array([
            ['2019-07-21T03:00:00.000', '2019-07-21T03:15:00.000'],
            ['2019-07-21T03:35:00.000', '2019-07-21T03:45:00.000'],
            [0, 0],
            [0, 0]])[2*NS + NC]
    elif PJnum == 22:
        tlim = np.array([
            ['2019-09-12T03:09:00.000', '2019-09-12T03:17:00.000'],
            [0, 0],
            [0, 0],
            [0, 0]])[2*NS + NC]
    elif PJnum == 26:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2020-04-10T15:00:00.000', '2020-04-10T15:15:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 28:
        tlim = np.array([
            ['2020-07-25T05:30:00.000', '2020-07-25T05:40:00.000'],
            ['2020-07-25T05:55:00.000', '2020-07-25T06:00:00.000'],
            ['2020-07-25T07:10:00.000', '2020-07-25T07:20:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 29:
        tlim = np.array([
            [0, 0],
            [0, 0],
            ['2020-09-16T03:15:00.000', '2020-09-16T03:45:00.000'],
            [0, 0]])[2*NS + NC]
    elif PJnum == 30:
        tlim = np.array([
            ['2020-11-08T01:10:00.000', '2020-11-08T01:30:00.000'],
            [0, 0],
            ['2020-11-08T02:45:00.000', '2020-11-08T03:10:00.000'],
            [0, 0]])[2*NS + NC]
    return tlim


def Time_Crossing_Io(PJnum, NS, NC):
    if PJnum == 5 and NS == 1 and NC == 0:
        return datetime(2017, 3, 27, 9, 30, 40), datetime(2017, 3, 27, 9, 31, 15)
    if PJnum == 7 and NS == 0 and NC == 0:
        return datetime(2017, 7, 11, 1, 24, 25), datetime(2017, 7, 11, 1, 25, 15)
    if PJnum == 13 and NS == 0 and NC == 0:
        return datetime(2018, 5, 24, 5, 13, 0), datetime(2018, 5, 24, 5, 13, 30)
    if PJnum == 22 and NS == 0 and NC == 0:
        return datetime(2019, 9, 12, 3, 18, 30), datetime(2019, 9, 12, 3, 19, 30)
    return 0


def Time_Crossing(name):
    PJnum = int(name[2:4])
    if name[4] == 'N':
        NS = 0
    else:
        NS = 1
    res = ' '
    if PJnum == 5 and NS == 0:
        res = "2017-03-27 08:32:36 - 08:35:41"
    elif PJnum == 5 and NS == 1:
        res = "2017-03-27 09:28:51 - 09:33:02"
    # elif PJnum == 6 and NS == 0:
    #     res = "2017-05-19 05:37:28 - 05:41:48"
    # elif PJnum == 12 and NS == 0:
    #     res = "2018-04-01 09:13:42 - 09:17:46"
    elif PJnum == 14 and NS == 0:
        res = "2018-07-16 04:47:50 - 04:52:12"
    # elif PJnum == 15 and NS == 0:
    #     res = "2018-09-07 00:47:14 - 00:50:27"
    elif PJnum == 20 and NS == 0:
        res = "2019-05-29 07:35:14 - 07:39:32"
    # elif PJnum == 23 and NS == 1:
    #     res = "2019-11-03 23:16:35 - 23:20:42"
    elif PJnum == 29 and NS == 0:
        res = "2020-09-16 01:58:29 - 02:01:45"
    # else: #PJnum == 30 and NS == 1:
    #     res = "2020-11-08 02:52:59 - 02:57:04"
    start = datetime.strptime(res[:19], "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(res[:11] + res[20:], "%Y-%m-%d - %H:%M:%S")
    return start, end


def Time_MainOval(PJnum, NS, NC):
    if PJnum == 1 and NS == 0 and NC == 0:
        return datetime(2016, 8, 27, 12, 11, 15)
    if PJnum == 1 and NS == 1 and NC == 0:
        return datetime(2016, 8, 27, 13, 35, 10)
    if PJnum == 3 and NS == 0 and NC == 0:
        return datetime(2016, 12, 11, 16, 24, 10)
    if PJnum == 3 and NS == 1 and NC == 0:
        return datetime(2016, 12, 11, 17, 37, 25)
    # if PJnum == 4 and NS == 0 and NC == 0:
    #     return datetime(2017, 2, 2, 12, 24, 0)
    if PJnum == 4 and NS == 1 and NC == 0:
        return datetime(2017, 2, 2, 13, 39, 0)
    if PJnum == 5 and NS == 1 and NC == 0:
        return datetime(2017, 3, 27, 9, 45, 40)
    if PJnum == 6 and NS == 1 and NC == 0:
        return datetime(2017, 5, 19, 6, 53, 20)
    if PJnum == 7 and NS == 0 and NC == 0:
        return datetime(2017, 7, 11, 1, 15, 25)
    if PJnum == 7 and NS == 1 and NC == 0:
        return datetime(2017, 7, 11, 2, 34, 25)
    if PJnum == 8 and NS == 1 and NC == 0:
        return datetime(2017, 9, 1, 22, 27, 50)
    if PJnum == 9 and NS == 0 and NC == 0:
        return datetime(2017, 10, 24, 17, 20, 20)
    if PJnum == 9 and NS == 1 and NC == 0:
        return datetime(2017, 10, 24, 18, 35, 00)
    if PJnum == 11 and NS == 0 and NC == 0:
        return datetime(2018, 2, 7, 12, 57, 22)
    if PJnum == 11 and NS == 1 and NC == 0:
        return datetime(2018, 2, 7, 14, 45, 10)
    if PJnum == 12 and NS == 0 and NC == 0:
        return datetime(2018, 4, 1, 9, 6, 48)
    if PJnum == 13 and NS == 0 and NC == 0:
        return datetime(2018, 5, 24, 5, 6, 5)
    if PJnum == 13 and NS == 1 and NC == 0:
        return datetime(2018, 5, 24, 6, 20, 30)
    if PJnum == 14 and NS == 0 and NC == 0:
        return datetime(2018, 7, 16, 4, 41, 20)
    if PJnum == 14 and NS == 1 and NC == 0:
        return datetime(2018, 7, 16, 6, 11, 40)
    if PJnum == 16 and NS == 0 and NC == 0:
        return datetime(2018, 10, 29, 20, 18, 40)
    if PJnum == 16 and NS == 1 and NC == 0:
        return datetime(2018, 10, 29, 21, 58, 5)
    # if PJnum == 20 and NS == 0 and NC == 0:
    #     return datetime(2019, 5, 29, 6, 39, 55)
    # if PJnum == 21 and NS == 0 and NC == 0:
    #     return datetime(2019, 7, 21, 3, 8, 20)
    if PJnum == 21 and NS == 0 and NC == 1:
        return datetime(2019, 7, 21, 3, 39, 45)
    if PJnum == 22 and NS == 0 and NC == 0:
        return datetime(2019, 9, 12, 3, 13, 35)
    return 0


def Get_ThetaTrajectory(PJnum, NS, NC):
    # [Get the theta: Juno FP trajectories to the local main oval]
    theta = np.zeros((31, 4))
    theta[1, :] = [80.50, 0, 48.77, 0]  # degs
    theta[3, :] = [74.60, 0, 88.57, 0]
    theta[4, :] = [81.59, 0, 81.66, 0]
    theta[5, :] = [86.38, 0, 85.52, 0]
    theta[6, :] = [21.11, 0, 61.49, 0]
    theta[7, :] = [87.66, 0, 58.27, 0]
    theta[8, :] = [89.70, 0, 88.35, 0]
    theta[9, :] = [83.15, 0, 86.84, 0]
    theta[11, :] = [30, 90, 80, 0]
    theta[12, :] = [20, 0, 0, 0]
    theta[13, :] = [90, 0, 80, 0]
    theta[14, :] = [80, 0, 30, 0]
    theta[16, :] = [80, 70, 80, 0]
    theta[18, :] = [80, 70, 80, 0]
    theta[20, :] = [20, 0, 0, 0]
    theta[21, :] = [60, 80, 0, 0]
    theta[22, :] = [45, 0, 0, 0]
    theta[26, :] = [0, 0, 20, 0]
    theta[28, :] = [60, 90, 80, 0]
    theta[29, :] = [0, 0, 60, 0]
    theta[30, :] = [45, 0, 80, 0]
    if NC == 3:
        return theta[PJnum, 2*NS]
    return theta[PJnum, 2*NS+NC]


def limits_ThetaFP(PJnum, NS, NC):
    # if sat:
    #     if PJnum == 5 and NS == 0 and NC == 0:
    #         return [(29.5, 31.5)]
    #     if PJnum == 5 and NS == 1 and NC == 0:
    #         return [(155, 157)]
    #     if PJnum == 14 and NS == 0 and NC == 0:
    #         return [(16, 19)]
    #     if PJnum == 20 and NS == 0 and NC == 0:
    #         return [(10, 14)]
    #     if PJnum == 29 and NS == 0 and NC == 0:
    #         return [(33, 35.5)]
    #     return [(0, 180)]
    if PJnum == 1 and NS == 0 and NC == 0:
        return [(8.5, 16)]
    if PJnum == 1 and NS == 1 and NC == 0:
        return [(158, 167.5)]
    if PJnum == 3 and NS == 0 and NC == 0:
        return [(8, 12)]
    if PJnum == 3 and NS == 1 and NC == 0:
        return [(168, 170.5)]
    if PJnum == 4 and NS == 0 and NC == 0:
        return [(8, 10), (11, 13)]
    if PJnum == 4 and NS == 1 and NC == 0:
        return [(164, 167.5)]
    if PJnum == 5 and NS == 1 and NC == 0:
        return [(164, 168)]
    if PJnum == 5 and NS == 1 and NC == 3:
        return [(155.3, 156.8)]
    if PJnum == 6 and NS == 1 and NC == 0:
        return [(163, 165), (166, 167.1)]
    if PJnum == 7 and NS == 0 and NC == 0:
        return [(5.5, 9), (11, 15)]
    if PJnum == 7 and NS == 1 and NC == 0:
        return [(158, 161), (163, 165)]
    if PJnum == 8 and NS == 1 and NC == 0:
        return [(169, 171)]
    if PJnum == 9 and NS == 0 and NC == 0:
        return [(14.5, 16.5), (17.5, 19)]
    if PJnum == 9 and NS == 1 and NC == 0:
        return [(160, 168)]
    if PJnum == 11 and NS == 0 and NC == 0:
        return [(9, 10.5)]
    if PJnum == 11 and NS == 1 and NC == 0:
        return [(163.5, 164.5), (165, 166.5)]
    if PJnum == 12 and NS == 0 and NC == 0:
        return [(6.8, 8)]
    if PJnum == 13 and NS == 0 and NC == 0:
        return [(7.5, 15)]
    if PJnum == 13 and NS == 0 and NC == 3:
        return [(19.5, 22)]
    if PJnum == 13 and NS == 1 and NC == 0:
        return [(159.5, 162), (165, 166)]
    if PJnum == 14 and NS == 0 and NC == 0:
        return [(7.5, 10.5)]
    if PJnum == 14 and NS == 1 and NC == 0:
        return [(162.5, 164)]
    if PJnum == 16 and NS == 0 and NC == 0:
        return [(12, 13.5)]
    if PJnum == 16 and NS == 1 and NC == 0:
        return [(163, 165.5), (166.5, 168.2)]
    if PJnum == 20 and NS == 0 and NC == 0:
        return [(26.8, 29.5)]
    if PJnum == 21 and NS == 0 and NC == 0:
        return [(17.8, 20)]
    if PJnum == 21 and NS == 0 and NC == 1:
        return [(10.5, 12), (12.8, 14.5)]
    if PJnum == 22 and NS == 0 and NC == 0:
        return [(9, 13)]
    if PJnum == 22 and NS == 0 and NC == 3:
        return [(20, 22)]
    if PJnum == 26 and NS == 1 and NC == 0:
        return [(163, 164)]
    if PJnum == 28 and NS == 0 and NC == 0:
        return [(17, 19)]
    if PJnum == 28 and NS == 0 and NC == 1:
        return [(15, 17)]
    if PJnum == 28 and NS == 1 and NC == 0:
        return [(166.5, 168.7)]
    return [(0, 180)]


def get_alpha(pj, ns, nc):
    if pj == 4 and ns == 1 and nc == 0:
        return 1.5
    if pj == 5 and ns == 1 and nc == 3:
        return 1
    if pj == 13 and ns == 0 and nc == 0:
        return 1
    if pj == 13 and ns == 0 and nc == 3:
        return 2
    if pj == 28 and ns == 0 and nc == 1:
        return 1
    if pj == 22 and ns == 0 and nc == 3:
        return 1.5
    return 3

def Get_MainOvalLongitude(PJnum, NS):
    # Obtained from visual observations on the data
    # Is improved then by the code
    theta = np.zeros((31, 2))
    theta[1, :] = [9.65, 163]
    theta[3, :] = [10, 169.45]
    theta[4, :] = [9.7, 166.3]
    theta[5, :] = [0, 166]
    theta[6, :] = [0, 164.65]
    theta[7, :] = [8.75, 164]
    theta[8, :] = [0, 169.5]
    theta[9, :] = [16.37, 166]
    theta[11, :] = [9.8, 164.7]
    theta[12, :] = [7.155, 0]
    theta[13, :] = [11.13, 0]
    theta[14, :] = [9, 163.55]
    theta[16, :] = [0, 166.845]
    # theta[20, :] = [26.864, 0]
    theta[20, :] = [27.75, 0]
    theta[21, :] = [11.99, 0]
    theta[22, :] = [11.62, 0]
    return theta[PJnum, NS]

def Get_yLim_Waves(PJnum, NS, NC):
    if PJnum == 1 and NS == 0 and NC == 0:
        return (3.5, 5)
    if PJnum == 1 and NS == 1 and NC == 0:
        return (5, 7) 
        # or whatever, no data for this crossing
    if PJnum == 3 and NS == 0 and NC == 0:
        return (4, 6)
    if PJnum == 3 and NS == 1 and NC == 0:
        return (5, 7)
    if PJnum == 4 and NS == 0 and NC == 0:
        return (5, 8)
    if PJnum == 4 and NS == 1 and NC == 0:
        return (4, 6)
    if PJnum == 5 and NS == 0 and NC == 0:
        return (2.5, 5)
    if PJnum == 5 and NS == 1 and NC == 0:
        return (2.5, 5)
    if PJnum == 5 and NS == 1 and NC == 3:
        return (4.5, 5.5)
    if PJnum == 6 and NS == 1 and NC == 0:
        return (2.5, 4.5)
    if PJnum == 7 and NS == 0 and NC == 0:
        return (4.5, 6.5)
    if PJnum == 7 and NS == 1 and NC == 0:
        return (4, 7)
    if PJnum == 8 and NS == 1 and NC == 0:
        return (4, 6)
    if PJnum == 9 and NS == 0 and NC == 0:
        return (12, 15)
    if PJnum == 9 and NS == 1 and NC == 0:
        return (2, 4)
    if PJnum == 11 and NS == 0 and NC == 0:
        return (2, 4)
    if PJnum == 11 and NS == 1 and NC == 0:
        return (2, 4)
    if PJnum == 12 and NS == 0 and NC == 0:
        return (4, 6)
    if PJnum == 13 and NS == 0 and NC == 0:
        return (5.7, 7)
    if PJnum == 13 and NS == 0 and NC == 3:
        return (7, 8.5)
    if PJnum == 13 and NS == 1 and NC == 0:
        return (4, 6)
    if PJnum == 14 and NS == 0 and NC == 0:
        return (5, 6.5)
    if PJnum == 14 and NS == 1 and NC == 0:
        return (2, 4)
    if PJnum == 16 and NS == 0 and NC == 0:
        return (2.5, 5)
    if PJnum == 16 and NS == 1 and NC == 0:
        return (2, 4)
    if PJnum == 20 and NS == 0 and NC == 0:
        return (0, 2)
    if PJnum == 21 and NS == 0 and NC == 0:
        return (2, 4)
    if PJnum == 21 and NS == 0 and NC == 1:
        return (8.5, 10.5)
    if PJnum == 22 and NS == 0 and NC == 0:
        return (7, 9)
    if PJnum == 22 and NS == 0 and NC == 3:
        return (8.5, 10)
    if PJnum == 26 and NS == 1 and NC == 0:
        return (1, 2)
    if PJnum == 28 and NS == 0 and NC == 0:
        return (4, 6)
    if PJnum == 28 and NS == 0 and NC == 1:
        return (15, 17)
    if PJnum == 28 and NS == 1 and NC == 0:
        return (1.7, 3)


if __name__ == '__main__':
    pass
