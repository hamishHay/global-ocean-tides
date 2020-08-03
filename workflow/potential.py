
from planetPy.constants import G
import numpy as np
from scipy.special import comb
from math import factorial

# def tidal_potential(forcing, body, lons, lats, time, order=1, kwargs={}):
#     forcing = forcing.lower()
#
#
#     if forcing is 'eccentricity':
#         potential_ecc()
#
#     elif forcing is 'obliquity':
#         potential_obl()


    #TODO - add all of these and then add to planetPy!!!!

def nk(n, k):
    if n<0:
        return (-1)**k * comb(abs(n) + k - 1, k)
    elif n==0 and k==0:
        return 1
    elif k <= n:
        return (-1)**(n-k) * float(comb(-k-1, n-k, exact=True))
    else:
        return 0


def kaula_G(l, p, q, e):
    B = e / (1 + np.sqrt(1 - e**2))

    k = np.arange(0, 10+1, 1)

    # print(p, l/2, p<=l/2)
    if p <= l/2:
        pd = p
        qd = q
    else:
        pd = l - p
        qd = -q

    P = np.zeros(len(k))
    Q = np.zeros(len(k))
    for i in range(len(k)):
        if qd > 0:
            h = k[i] + qd
            n = k[i]
        else:
            h = k[i]
            n = k[i] - qd

        for r in range(0, h+1):
            P[i] += nk(2*pd - 2*l, h - r) * ((-1)**r)/factorial(r) * \
                    ((l - 2*pd + qd)*e / (2*B))**r

        for r in range(0, n+1):
            Q[i] += nk(-2*pd, n - r) / factorial(r) * \
                    ( (l - 2*pd + qd)*e / (2*B) )**r


    Ge = (-1)**(abs(q)) * (1 + B**2.0)**l * B**(abs(q))
    Ge *= np.sum(P * Q * B**(2*k))

    return Ge

def kaula_F(l, m, p, i=0.0):

    # if p <= (l-m)/2:
    #     tmax = p
    # if p >= (l-m)/2 and (l-m)%2 == 0:
    #     k = (l - m)/2
    # elif p >= (l-m)/2 and (l-m)%2 != 0:
    #     k = (l - m - 1)/2

    k, d = divmod((l-m)/2, 1)

    tmax = min(p, k)

    Fi = 0.0
    for t in range(0, int(tmax)+1):
        ssum = 0
        for s in range(0, m+1):
            cmin = 0
            if m+t <= p+s:
                cmin = p - t - m + s

            cmax = l - m - 2*t + s
            if m+p+t <= l+s:
                cmax = p - t


            c = np.arange(cmin, cmax+1, 1)

            # if (l-m)%2 == 0:
            #     cp = c - (l-m)/2
            # else:
            #     cp = c - (l-m-1)/2

            c = comb(l - m - 2*t + s, c) * comb(m - s, p - t - c) * (-1)**(c-k)


            ssum +=  comb(m, s) * np.cos(i)**s * np.sum(c)

        Fi += factorial(2*l - 2*t) * np.sin(i)**(l-m-2*t) / ( factorial(t) * factorial(l - t) * factorial(l-m-2*t) * 2**(2*l - 2*t)) * ssum

    return Fi

def Plm(l, m, colat):

    if l==2:
        if m==0:
            P = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
        elif m==1:
            P = -3*np.cos(colat)*np.sin(colat)
        elif m==2:
            P = 3.*(1. - np.cos(colat)**2.0)
    elif l==3:
        if m==0:
            P = 0.5*(5*np.cos(colat)**3.0 - 3*np.cos(colat))
        

    return P



def potential_kaula(l, m, q, M, a, R, n, e, i, t, colat=0.0, lon=0.0):
    Ulmq = colat.copy()*0.0

    B = G*M * factorial(l-m)/factorial(l+m)
    if m != 0:
        B *= 2

    for p in range(0, l+1):
        C = 1.0/a**(l+1) * kaula_F(l, m, p, i)

        vlmpq = (l - 2*p + q)*n*t
        vlmpq_n = (l - 2*p - q)*n*t
        if (l-m)%2 == 0:
            trig_q  = np.cos( vlmpq - m*(lon + n*t) )
            trig_qn = np.cos( vlmpq_n - m*(lon + n*t) )
        else:
            trig_q = np.sin( vlmpq - m*(lon + n*t) )
            trig_qn = np.sin( vlmpq_n - m*(lon + n*t) )

        if q != 0:
            C *= (kaula_G(l, p, q, e)*trig_q + kaula_G(l, p, -q, e)*trig_qn)
        else:
            C *= kaula_G(l, p, q, e)*trig_q

        Ulmq += B * C * R**l * Plm(l,m,colat)
        # if l==2 and m==1:# and p==0 and q==0:
        #     print(p, q, m, kaula_F(l, m, p, i), kaula_G(l, p, q, e), 2*factorial(l-m)/factorial(l+m) * kaula_F(l, m, p, i) * kaula_G(l, p, q, e))

    return Ulmq



def potential_moonmoon(a1, a2, m2, R1, n1, n2, omega1, t, colat=0.0, lon=0.0):

    df = (n1 - n2)*t
    # print(n1-n2)
    p = a1**2 + a2**2 - 2*a1*a2*np.cos(df)
    r = np.sqrt(a1**2 + a2**2 - 2*a1*a2*np.cos(df))
    Mag = 0.5 * G * m2 * (R1 / p)**2.0 * 1./np.sqrt(p)

    # Mag = 1.0
    sin_lon_12 = a2*np.sin(df)
    cos_lon_12 = (a1 - a2*np.cos(df))

    Ut = (np.cos(lon)*cos_lon_12 + np.sin(lon)*sin_lon_12)**2.0

    Ut = Mag*(3*np.sin(colat)**2.0 * Ut - p)
    return Ut


def potential_moonmoon_dt(a1, a2, m2, R1, n1, n2, omega1, t, colat=0.0, lon=0.0):
    df = (n1 - n2)*t
    n = n1 - n2
    Mag = G * m2 * R1**2.0
    r = np.sqrt(a1**2 + a2**2 - 2*a1*a2*np.cos(df))

    cos_df = np.cos(df)
    sin_df = np.sin(df)

    # Mag = 1.0
    sin_l = a2*np.sin(df)/r
    cos_l = (a1 - a2*np.cos(df))/r

    sin_l_dt = a2*n/r**3.0 * (r**2.*cos_df - a1*a2*sin_df**2.)
    cos_l_dt = a2*n*sin_df/r**3.0 * (r**2. - a1*(a1 - a2*cos_df))

    sin_clat = np.sin(colat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    r_dt = a1*a2*n*sin_df/r

    Ut = -(3*sin_clat**2.0 * (cos_lon*cos_l + sin_lon*sin_l)**2.0 - 1)*3*r**2.0 * r_dt
    Ut += r**3.0 * 6*sin_clat**2.0*(cos_lon*cos_l + sin_lon*sin_l)*(cos_lon*cos_l_dt + sin_lon*sin_l_dt)
    Ut *= 0.5*Mag / r**6.

    return Ut

def potential_ecc(e, R1, n1, t, colat, lon, sync=False):
    # if sync:
    #     omega1 = n1

    cosM = np.cos(n1*t)
    sinM = np.sin(n1*t)

    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)

    U = n1**2.0 * R1**2.0 * e * (-1.5*P20*cosM + 1./8.*P22*(7.*np.cos(2.*lon - n1*t) - np.cos(2*lon + n1*t)))
    return U

# def potential_ecc3(e, R1, n1, t, colat, lon, sync=False):


def potential_obl(o, R1, n1, t, colat, lon, sync=False):
    # if sync:
    #     omega1 = n1

    cosM = np.cos(n1*t)
    sinM = np.sin(n1*t)

    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)


    M = n1*t
    U = n1**2.0 * R1**2.0 * o * 0.5*(P21*(np.cos(lon-M) + np.cos(lon+M)))

    return U

def potential_ecc_lib_east(e, R1, n1, t, colat, lon, sync=False):
    # if sync:
    #     omega1 = n1

    cosM = np.cos(n1*t)
    sinM = np.sin(n1*t)

    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)

    U = n1**2.0 * R1**2.0 * e * (1./8.*P22*(7.*np.cos(2.*lon - n1*t))) # East
    return U

def potential_ecc_lib_west(e, R1, n1, t, colat, lon, sync=False):
    # if sync:
    #     omega1 = n1

    cosM = np.cos(n1*t)
    sinM = np.sin(n1*t)

    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)

    U = n1**2.0 * R1**2.0 * e * (1./8.*P22*(-np.cos(2*lon + n1*t))) # West
    return U

def potential_ecc_lib(e, R1, n1, t, colat, lon, sync=False):
    # if sync:
    #     omega1 = n1

    cosM = np.cos(n1*t)
    sinM = np.sin(n1*t)

    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)

    U = n1**2.0 * R1**2.0 * e * (1./8.*P22*(-np.cos(2*lon + n1*t))) # West
    U += n1**2.0 * R1**2.0 * e * (1./8.*P22*(7.*np.cos(2.*lon - n1*t))) # East
    return U

def potential_ecc_dt(e, R1, n1, t, colat, lon, sync=False):
    P20 = 0.5*(3.*np.cos(colat)**2.0 - 1.0)
    P22 = 3.*(1. - np.cos(colat)**2.0)
    P21 = -3*np.cos(colat)*np.sin(colat)

    U = n1**2.0 * R1**2.0 * e * n1 * (1.5*P20*np.sin(n1*t) + 1./8.*P22*(7.*np.sin(2.*lon - n1*t) + np.sin(2*lon + n1*t)))

    return U


if __name__ == '__main__':
    # print(kaula_F(2, 0, 1, 0.0))
    for p in range(0, 2+1):
        for q in range(-2, 2+1):
            print(kaula_G(2, p, q, 0.0000000000001))
            # potential_kaula()
            # print(p, q, kaula_G(2, p, q, 0.00001))
