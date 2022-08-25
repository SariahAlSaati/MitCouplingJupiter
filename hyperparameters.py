# this file contain all the hyperparameters that are used in this project
# Feel free to modify them to adapt them to your use

# paths, replace antislash to slash if in Windows
pathtoMIT = "./"
pathtodata = pathtoMIT + "Data/"
pathtoinstruments = pathtoMIT + "Data/instruments/"
pathtoresults = pathtoMIT + "Results/"
pathtoatmo = pathtoresults + "MODatmo/"
pathtoiono = pathtoresults + "MODionos/"
pathtoelec = pathtoresults + "MODelectro/"
pathtoresB = pathtoresults + "MAGresiduals/"
pathtoplots = pathtoresults + "PLT/"


# file electrodynamics.py
resolution = 2      # time resolution in seconds
Offset_min = [0, 0]  # time range: 0min ahead and late than standard one.
KK = 0.3            # slippage from Tao et al. (2009# 2010)
Rj = 71492.0  # km
smoothingWindow = 120  # seconds


NS = ['N', 'S']
NSlong = ['North', 'South']
