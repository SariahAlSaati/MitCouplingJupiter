import UTLutils as ut
import numpy as np
from scipy.special import gammaln
from scipy.special import lpmn


def CAN81(rho_in, z_in):
    # Description:
    #   Jupiter's external magnetic field model:
    #   The Connerney+1981_JGR model: Modeling the Jovian Current Sheet and Inner Magnetophere.
    #   The CAN model should be only used near the Jupiter (<~30 RJ).
    #
    # Input:
    #   rho and z in cylindrical coords, z is along the magnetic axis.
    #
    # Output:
    #   Brho and Bz (nT) also in cylindrical coords. Note that the model is axis-symmetric.
    #
    # History:
    #   Writen by Yuxian Wang 2019-12-07 23:41:16
    rho = float(rho_in)
    z = float(z_in)

    # [constants]
    D = 2.5          # Rj #Thickness of the disk
    a = 5.0           # Rj #R_0 inner radius of disk
    b = 50.0  # 70.0         # Rj         # 50.0 in Connerney1981 #R_1 outer radius of disk
    mu0_I0 = 450.0  # 175.0*2       # 225.0*2 in Connerney1981

    minval = 1.0e-15

    # [Approximate formulas given in Connerney+1981_The magnetic field in Jupiter.]
    # [Cylindrical coordinates: in nT]
    if (rho <= a):
        F1 = np.sqrt((z-D)**2 + a**2)
        F2 = np.sqrt((z+D)**2 + a**2)
        F3 = np.sqrt(z**2 + a**2)

        Brho = 0.5*rho*(1.0/F1 - 1.0/F2)
        tmp = (z - D)/F1**3 - (z + D)/F2**3
        Bz = 2.0*D/F3 - 0.25*rho**2*tmp
    else:
        F1 = np.sqrt((z-D)**2 + rho**2)
        F2 = np.sqrt((z+D)**2 + rho**2)
        F3 = np.sqrt(z**2 + rho**2)

        Brho = (F1-F2+2*D)/rho
        if (abs(z) >= D) and (z < 0):
            Brho = (F1-F2-2*D)/rho
        if (abs(z) < D):
            Brho = (F1-F2+2*z)/rho
        Brho -= 0.25*a**2*rho*(1.0/F1**3 - 1.0/F2**3)
        tmp = (z-D)/F1**3 - (z+D)/F2**3
        Bz = 2.0*D/F3 - 0.25*a**2*tmp

    F1 = np.sqrt((z-D)**2 + b**2)
    F2 = np.sqrt((z+D)**2 + b**2)
    F3 = np.sqrt(z**2 + b**2)
    Brho2 = 0.5*rho*(1/F1 - 1/F2)
    Bz2 = 2*D/F3 - 0.25*rho**2*((z-D)/F1**3 - (z+D)/F2**3)

    Brho -= Brho2
    Bz -= Bz2

    Brho *= 0.5*mu0_I0
    Bz *= 0.5*mu0_I0

    ans = [Brho, Bz]
    return ans


def Coord_JSM2SIII(coordin):
    # Transform from Jupiter Magnetic Coordinate (x, y, z) to System III coordinate (x, y, z)
    # More details about the Magnetic coordinate:
    #   Jupiter Solar Magnetospheric (JSM) coordinates Coordinates used during the Galileo mission,
    #   which used a dipole tilt of 9.6 degrees and lamda_III = 202 degrees based on the O4 model
    #   of Connerney (1981). Therefore this coordinates are special for CAN current model.
    #
    # More details about the System III coordinate:
    # Here refers to right-hand System III (S3RH).

    # (1+2) Rz#Ry
    Rzy = np.transpose(np.array(
        [
            [-0.91419964974697621,      0.36936050381396512,     -0.16676875948905945],
            [-0.37460649013519287,     -0.92718392610549927,      0.00000000000000000],
            [-0.15462531317480988,     0.062472659656396701,      0.98599600791931152]
        ]
    ))
    coordout = np.matmul(Rzy, np.array(coordin))
    return coordout


def Coord_SIII2JRM09(coordin):
    # Transform fromSystem III coordinate (x, y, z) to  Jupiter Magnetic Coordinate (x, y, z)
    # More details about the Magnetic coordinate:
    #   JRM09 model during Juno mission,
    #   which used a dipole tilt of 10.31 degrees and lamda_III = 196.61 degrees based on the JRM09 model
    #   of Connerney (2018).
    #
    # More details about the System III coordinate:
    #   Here refers to right-hand System III (S3RH).

    # (1+2) Ry#Rz#coordin
    Ryz = np.transpose(np.array(
        [
            [-0.94280024922193206,     -0.28585568070411682,     -0.17150585706493260],
            [0.28124020256790061,     -0.95827269554138184,    0.051160722562737515],
            [-0.17897395789623260,      0.00000000000000000,      0.98385381698608398]
        ]
    ))

    coordout = np.matmul(Ryz, np.array(coordin))
    return coordout


def Coord_SIII2JSM(coordin):
    # (INVERSE TRANSFORMATION)
    # Transform from Jupiter Magnetic Coordinate (x, y, z) to System III coordinate (x, y, z)
    # More details about the Magnetic coordinate:
    #   Jupiter Solar Magnetospheric (JSM) coordinates Coordinates used during the Galileo mission,
    #   which used a dipole tilt of 9.6 degrees and lamda_III = 202 degrees based on the O4 model
    #   of Connerney (1981). Therefore this coordinates are special for CAN current model.
    #
    # More details about the System III coordinate:
    #   Here refers to right-hand System III (S3RH).
    # Written by Yuxian

    # (1+2) Ry#Rz#coordin
    Ryz = np.transpose(np.array(
        [
            [-0.91419964974697621,     -0.37460649013519287,     -0.15462531317480988],
            [0.36936050381396512,     -0.92718392610549927,     0.062472659656396701],
            [-0.16676875948905945,      0.00000000000000000,      0.98599600791931152]
        ]
    ))

    coordout = np.matmul(Ryz, np.array(coordin))
    return coordout


def Get_Schmdt_Coeff():
    # The coefficients are from Connerney et al. (2018).

    GSMDT = np.zeros((11, 11))
    HSMDT = np.zeros((11, 11))

    GSMDT[0] = [0.0000000000000000, 410244.7000000000, 11670.40000000000, 4018.600000000000, -34645.40000000000, -
                18023.60000000000, -20819.60000000000, 598.4000000000000, 10059.20000000000, 9671.799999999999, -2299.500000000000]
    GSMDT[1] = [0.0000000000000000, -71498.30000000000, -56835.80000000000, -37791.10000000000, -8247.600000000000,
                4683.900000000000, 9992.900000000000, 4665.900000000000, 1934.400000000000, -3046.200000000000, 2009.700000000000]
    GSMDT[2] = [0.0000000000000000, 0.0000000000000000, 48689.50000000000, 15926.30000000000, -2406.100000000000,
                16160.00000000000, 11791.80000000000, -6495.700000000000, -6702.900000000000, 260.9000000000000, 2127.800000000000]
    GSMDT[3] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -2710.500000000000, -11083.80000000000, -
                16402.00000000000, -12574.70000000000, -2516.500000000000, 153.7000000000000, 2071.300000000000, 3498.300000000000]
    GSMDT[4] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -17837.20000000000, -
                2600.700000000000, 2669.700000000000, -6448.500000000000, -4124.200000000000, 3329.600000000000, 2967.600000000000]
    GSMDT[5] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -
                3660.700000000000, 1113.200000000000, 1855.300000000000, -867.2000000000001, -2523.100000000000, 16.30000000000000]
    GSMDT[6] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 7584.900000000000, -2892.900000000000, -3740.600000000000, 1787.100000000000, 1806.500000000000]
    GSMDT[7] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 2968.000000000000, -732.4000000000000, -1148.200000000000, -46.50000000000000]
    GSMDT[8] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -2433.200000000000, 1276.500000000000, 2897.800000000000]
    GSMDT[9] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -1976.800000000000, 574.5000000000000]
    GSMDT[10] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1298.900000000000]

    HSMDT[0] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
    HSMDT[1] = [0.0000000000000000, 21330.50000000000, -42027.30000000000, -32957.30000000000, 31994.50000000000,
                45347.90000000000, 14533.10000000000, -7626.300000000000, -2409.700000000000, -8467.400000000000, -4692.600000000000]
    HSMDT[2] = [0.0000000000000000, 0.0000000000000000, 19353.20000000000, 42084.50000000000, 27811.20000000000, -
                749.0000000000000, -10592.90000000000, -10948.40000000000, -11614.60000000000, -1383.800000000000, 4445.800000000000]
    HSMDT[3] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -27544.20000000000, -926.1000000000000,
                6268.500000000000, 568.6000000000000, 2633.300000000000, 9287.000000000000, 5697.700000000000, -2378.600000000000]
    HSMDT[4] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 367.1000000000000,
                10859.60000000000, 12871.70000000000, 5394.200000000000, -911.9000000000000, -2056.300000000000, -2204.300000000000]
    HSMDT[5] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                9608.400000000000, -4147.800000000000, -6050.800000000000, 2754.500000000000, 3081.500000000000, 164.1000000000000]
    HSMDT[6] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 3604.400000000000, -1526.000000000000, -2446.100000000000, -721.2000000000001, -1361.600000000000]
    HSMDT[7] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, -5684.200000000000, 1207.300000000000, 1352.500000000000, -2031.500000000000]
    HSMDT[8] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -2887.300000000000, -210.1000000000000, 1411.800000000000]
    HSMDT[9] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1567.600000000000, -714.3000000000000]
    HSMDT[10] = [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1676.500000000000]
    return GSMDT, HSMDT


def Safe0Division(numerator, denominator, valueifboth0=None):
    # This helper function divides one array into another, while preventing any "divide by zero" errors that might result:

    # Do a "safe" division of numerator by denominator, without a divide by zero.
    # Both arguments are assumed to have same number of elements.
    # Where both the numerator and denomiator are zero, the result is normally returned as 1.,
    #  unless the keyword VALUEBOTH0 is set to a different number, such as zero.
    # Otherwise, the result is returned as near infinity (1.e36), with same sign
    # as numerator.  Since some calculations, such as trigometric functions like
    # COS(pi/2) result in very small numbers ~4.37e-8 instead of exactly zero, then any
    # number in the numerator that is smaller than 1.e-8 will be treated like zero,
    # This value is slightly larger than cos(!PI/2)

    numerator = np.array(numerator)
    denominator = np.array(denominator)
    n = len(numerator)
    nearzero = 1e-8
    res = np.zeros(n)

    if not valueifboth0:
        valueifboth0 = 1.

    for i in range(n):
        if np.abs(denominator[i]) < nearzero:
            if np.abs(numerator[i]) < nearzero:
                res[i] = valueifboth0
            else:
                res[i] = np.sign(numerator[i])*np.inf
        else:
            res[i] = numerator[i] / denominator[i]
    return res


def AllLegendre(x, lmax, mmax, NORMALIZE=1, SinTheta=None):
    # If doing the derivatives and X contains cos(theta),
    #   then you should set the SinTheta keyword parameter to non-zero,
    #   so that the result contains the product of -sin(theta) and dP(cos(theta))/dtheta.
    #   If any of the angles theta are negative (so that sin(theta) would be negative),
    #   then you should provide an array with the Sin of the angle theta in the SinTheta keyword parameter.

    # The default behavior is to apply a Schmidt quasi-normalized  factor.
    # Set the keyword NORMALIZE to 0 to skip the normalization.
    """Return an array of size (n, mmax + 1, lmax + 1)."""
    n = len(x)
    Plm = np.zeros((n, mmax+1, lmax+1))
    dPlm = np.zeros((n, mmax+1, lmax+1))
    for i in range(n):
        Plm[i], dPlm[i] = lpmn(mmax, lmax, x[i])
        if NORMALIZE:
            for l in range(lmax + 1):
                for m in range(1, min(mmax + 1, l + 1)):
                    # If m is odd, then multiply by -1
                    if (m % 2):
                        asign = -1.
                    else:
                        asign = 1
                    # apply the Schmidt quasi-normalized function.
                    # nf=asign*SQRT( 2. *factorial(l-m)/factorial(l+m) )
                    nf = asign * \
                        np.sqrt(
                            2. * np.exp(gammaln(float(l-m+1)) - gammaln(float(l+m+1))))
                    # more accurate way to get (l-m)!/(l+m)!
                    Plm[i, m, l] = nf * Plm[i, m, l]
                    dPlm[i, m, l] = nf * dPlm[i, m, l]
        try:
            SinTheta
            # Set this keyword if X is Cosine(Theta), so need to multiply derivative by -sine(theta)

            # Set the keyword SinTheta as the sinus of theta in an array of the same size as x
        except NameError:
            easter = "do nothing"
        else:
            signs = -1 * np.sign(SinTheta)
            for l in range(lmax + 1):
                dPlm[i, 0, l] = signs[i] * SinTheta[i] * dPlm[i, 0, l]
                for m in range(1, min(mmax + 1, l + 1)):
                    dPlm[i, m, l] = signs[i] * SinTheta[i] * dPlm[i, m, l]

    return Plm, dPlm


def GetJupiterMag(r_in, theta_in, phi_in, Bxyz=None, CAN=None):
    # +
    # =================================================
    # Description:
    #    This program is to calculate the Jovian magnetic field of JRM09 (Connerney et al., 2018) and
    #    CAN (Connerney et al., 1981) model.
    #
    #    All of the relevant subroutines are included in this single file, including the Schmdt coefficients
    #    needed in JRM09 model.
    #
    #   (1) Jupiter's Magnetic field Model: JRM09 (+CAN), CAN.
    #   (2) Refer to: Connerney+2018_GRL  JRM09 model
    #                       Connerney+1981            CAN model
    #   (3) All the coordinates used here are Right-Hand-Side, because of SPICE convention.
    #   (4) The positions can be in Arrays, i.e., multi-positions at one time, which can be more efficient.
    #
    # Input:
    #   r (Rj), theta (deg), phi (deg), spherical coordinates in System III RHS (i.e., IAU_Jupiter), better use double!
    #
    #   optional:
    #       /CAN              include the CAN model or not (1 = yes)
    #
    # Output:
    #   Br (nT), Bt (nT), Bp (nT), Bmag (nT) in System III RHS.
    #   Bxyz = Bxyz   optional output Bxyz in System III RHS
    #
    # Example:
    #   GetJupiterMag, [20.0d0,20.00000001d0], [90.0d0,90.000001d0], [120.0d0, 120.0d0], Br, Bt, Bp, Bmag1, Bxyz = Bxyz, /can
    #   GetJupiterMag00, 20.0d0, 90.000001d0, 120.0d0, Br, Bt, Bp, Bmag2, Bxyz = Bxyz, /can
    #
    # History:
    #   (1) writen by Yuxian Wang 2020-01-16 02:26:49
    #   (2) Revise an error in Lines 581 - 587. by Yuxian 2020-07-31 16:07:59
    #   ywang@irap.omp.eu + yxwang@spaceweather.ac.cn
    # =================================================

    # [position in: ['Rj, deg, deg'] to ['Rj, rad, rad']]
    r = np.array(r_in)
    theta = np.array(theta_in)*np.pi/180
    phi = np.array(phi_in)*np.pi/180
    npoints = len(theta)
    Br = np.zeros(npoints)
    Bt = np.zeros(npoints)
    Bp = np.zeros(npoints)

    # [try to avoid invoke restore one more time.]
    GSMDT, HSMDT = Get_Schmdt_Coeff()
    maxN = 10    # JRM09 degrees

    # [Get the Legendre: dPlm is dPlmdtheta, x must less than < 1]
    x = np.cos(theta)
    lmax = maxN
    mmax = maxN
    sintheta = np.sin(theta)

    # here Plm and dPlm are arrays of size (n, mmax + 1, lmax + 1)
    Plm, dPlm = AllLegendre(x, lmax, mmax, NORMALIZE=1, SinTheta=sintheta)
    # [Calculate Components of B]
    for L in range(1, lmax + 1):
        radialfactor = r**(-(L+2.0))
        for M in range(min(mmax + 1, L+1)):
            # -------------find radial b field component---------------
            gsmdtlm = GSMDT[M, L]
            hsmdtlm = HSMDT[M, L]
            phifactor = gsmdtlm*np.cos(M*phi) + hsmdtlm*np.sin(M*phi)

            Pnm = Plm[:, M, L]
            Br += (L+1) * radialfactor * phifactor * Pnm

            # --------------find theta b field component---------------
            dpnmdtheta = dPlm[:, M, L]  # dpdtheta(L, M, theta, /inrad)
            Bt -= radialfactor * phifactor * dpnmdtheta

            # ----------------find phi b field component-------------------
            Bp += M * radialfactor * \
                (gsmdtlm*np.sin(M*phi) - hsmdtlm*np.cos(M*phi)) * Pnm

    # --------correct bphi and don't divide by 0!------------
    ir = np.where(sintheta != 0)[0]
    count = len(ir)
    if (count > 0):
        for i in ir:
            Bp[i] /= sintheta[i]

    ir = np.where(sintheta == 0)[0]
    count = len(ir)
    if (count > 0):
        for i in ir:
            Bp[i] = 0

    # [Add CAN model]
    if CAN:
        # [IAU -> JSM   xyz]
        IAUx, IAUy, IAUz = np.transpose(ut.Sph2CarP(r_in, theta_in, phi_in))
        JSMxyz = np.array([Coord_SIII2JSM(
            np.array([IAUx[i], IAUy[i], IAUz[i]])) for i in range(len(IAUx))])

        # [JSM: xyz -> rho z]

        rho = np.sqrt(JSMxyz[:, 0]**2 + JSMxyz[:, 1]**2)
        zzz = JSMxyz[:, 2]
        Bcan = np.zeros((npoints, 2))    # [rho z, point]
        for ip in range(npoints):
            Bcan[ip] = CAN81(rho[ip], zzz[ip])

        # [B_JSM: rho z -> xyz]
        # Bug fixed here. 2020-07-31 15:54:15
        phi = np.arctan2(JSMxyz[:, 1], JSMxyz[:, 0])
        # ir = where(JSMxyz[0, *] lt 0, count) & if (count gt 0) then phi += !dpi
        # ir = where(phi lt 0, count) & if (count gt 0) then phi += 2.0*!dpi
        JSMx = np.array(JSMxyz[:, 0])

        ir = np.where(phi < 0)[0]
        count = len(ir)
        if (count > 0):
            phi[ir] += 2.0*np.pi

        BJSMx = Bcan[:, 0]*np.cos(phi)
        BJSMy = Bcan[:, 0]*np.sin(phi)
        BJSMz = Bcan[:, 1]

        # [JSM to System III]
        BIAUxyz = np.array([Coord_JSM2SIII(
            np.array([BJSMx[i], BJSMy[i], BJSMz[i]])) for i in range(len(BJSMx))])
        IAUxyz = np.transpose(np.array([IAUx, IAUy, IAUz]))
        BIAUr, BIAUt, BIAUp = np.transpose(
            ut.Car2SphV(IAUxyz, BIAUxyz[:, 0], BIAUxyz[:, 1], BIAUxyz[:, 2]))

        Br += BIAUr
        Bt += BIAUt
        Bp += BIAUp

    Bmag = np.sqrt(Br**2 + Bt**2 + Bp**2)

    # [Bx, By, Bz: Bxyz in System III RHS]
    if Bxyz:
        posi_sph = np.transpose(np.array([r_in, theta_in, phi_in]))
        Bx, By, Bz = np.transpose(ut.Sph2CarV(posi_sph, Br, Bt, Bp))

    if Bxyz:
        return Br, Bt, Bp, Bmag, Bx, By, Bz
    return Br, Bt, Bp, Bmag


if __name__ == '__main__':
    pass
