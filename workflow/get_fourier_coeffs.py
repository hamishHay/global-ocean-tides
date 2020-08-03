import potential
import numpy as np
from pyshtools.expand import SHExpandDH
from pyshtools.expand import MakeGridPoint
from pyshtools.legendre import PlmBar, PLegendreA
import h5py
# import planet_data
import sys
import matplotlib.pyplot as plt
from planetPy.constants import AU, G
from planetPy.database import Database

database = Database().moon_dict

# grid spacing in latitiude and longitude for SH computation
dx = 2.0
dy = 2.0

# max spherical harmonic degree for calculation
LMAX = 2

# time spacing
dt = 0.01
FM = 200

def get_fourier_coefficients(time_series, real=True):
    if real:
        fft_coeffs = np.fft.rfft(time_series)/(time_series.size)
        fft_freqs = np.arange(0, fft_coeffs.size)

    else:
        fft_coeffs = np.fft.fft(time_series)/(time_series.size)
        fft_coeffs = np.fft.fftshift(fft_coeffs)
        fft_freqs = np.arange(-len(fft_coeffs)/2,len(fft_coeffs)/2, 1)

    return fft_coeffs, fft_freqs

def get_sh_coefficients(gridded_data):
    time_levels = np.shape(gridded_data)[-1]

    clmt = np.zeros((time_levels, 2, LMAX+1, LMAX+1))

    for k in range(time_levels):
        clmt[k] = SHExpandDH(gridded_data[:, :, k], sampling=2, norm=3, csphase=1, lmax_calc=LMAX)

    return clmt

def main():
    save_file = h5py.File('saturnian_coefficients.h5','w')
    planets = ['trappist-1 b', 'trappist-1 c', 'trappist-1 d', 'trappist-1 e',
               'trappist-1 f', 'trappist-1 g', 'trappist-1 h']
    forcings = [['trappist-1 c', 'trappist-1 d', 'eccentricity'],
                ['trappist-1 b', 'trappist-1 d', 'eccentricity'],
                ['trappist-1 c', 'trappist-1 e', 'eccentricity'],
                ['trappist-1 d', 'trappist-1 f', 'eccentricity'],
                ['trappist-1 e', 'trappist-1 g', 'eccentricity'],
                ['trappist-1 f', 'trappist-1 h', 'eccentricity'],
                ['trappist-1 g', 'eccentricity']]

    planets = ['mimas', 'enceladus', 'tethys', 'dione']
    forcings = [['enceladus'],
                ['mimas', 'tethys'],
                ['enceladus', 'dione'],
                ['tethys']]

    NP = len(planets)  # Number of planets

    real=True
    test=False

    FM=200

    # Create 1D arrays in colatitue, longitude, and time
    colats_ = np.radians(np.arange(0., 180., dy, dtype=np.float64))
    lons_   = np.radians(np.arange(0., 360., dx, dtype=np.float64))
    t_      = np.linspace(0., 1.0, FM, endpoint=False, dtype=np.float64)

    # Loop through each planet
    for j in range(NP):
        planet = planets[j]
        P_ID = j

        # Get planet P data:
        # a1   = planet_data.p_a[P_ID]
        # R1   = planet_data.p_R[P_ID]
        # n1   = planet_data.p_n[P_ID]
        # e1   = planet_data.p_e[P_ID]
        # o1   = planet_data.p_o[P_ID]

        a1 = database[planet.capitalize()].semimajor_axis
        R1 = database[planet.capitalize()].radius
        n1 = database[planet.capitalize()].mean_motion
        e1 = database[planet.capitalize()].eccentricity
        # o1 = database[planet.capitalize()].obliquity

        # Create group for tides raised on planet j
        grp = save_file.create_group(planet)

        header = '  '
        # Loop through each tide raising planet
        for i in range(len(forcings[j])):
            forcing = forcings[j][i]
            # Create subgroup for the tide raising planet i
            print(planet, forcing)
            subgrp = grp.create_group(forcing)

            if forcing is not 'eccentricity' and forcing is not 'obliquity':
                # Get planet i data
                a2 = database[forcing.capitalize()].semimajor_axis
                m2 = database[forcing.capitalize()].mass
                n2 = database[forcing.capitalize()].mean_motion

                # Get conjuction/forcing period
                P = 2*abs(np.pi/(n1 - n2))

                # Create 3D grid in longitude, colatitude, and time
                lons, colats, t = np.meshgrid(lons_, colats_, t_*P)

                # Reverse Semimajor axes if tide raiser is inside or outside
                if a2 > a1:
                    U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t, colats, lons)
                else:
                    U = potential.potential_moonmoon(a2, a1, m2, R1, n2, n1, n2, t, colats, lons)


            else:
                P = 2*abs(np.pi/n1)
                lons, colats, t = np.meshgrid(lons_, colats_, t_*P)
                if forcing is "eccentricity":
                    U = potential.potential_ecc(e1, R1, n1, t, colats, lons)

                else:
                    U = potential.potential_obl(o1, R1, n1, t, colats, lons)

            clmt = get_sh_coefficients(U)

            C20 = clmt[:,0,2,0]
            C22 = clmt[:,0,2,2]
            C21 = clmt[:,0,2,1]
            S20 = clmt[:,1,2,0]
            S22 = clmt[:,1,2,2]
            S21 = clmt[:,1,2,1]

            C20f, freqs = get_fourier_coefficients(C20, real=real)
            C22f, freqs = get_fourier_coefficients(C22, real=real)
            C21f, freqs = get_fourier_coefficients(C21, real=real)
            S20f, freqs = get_fourier_coefficients(S20, real=real)
            S22f, freqs = get_fourier_coefficients(S22, real=real)
            S21f, freqs = get_fourier_coefficients(S21, real=real)

            C20f = 2.*C20f[:]
            C22f = 2.*C22f[:]
            C21f = 2.*C21f[:]
            S20f = 2.*S20f[:]
            S22f = 2.*S22f[:]
            S21f = 2.*S21f[:]
            freqs = freqs[:]

            FM = len(C20f)

            C20a = C20f.real
            C20b = C20f.imag
            C22a = C22f.real
            C22b = C22f.imag
            C21a = C21f.real
            C21b = C21f.imag
            S20a = S20f.real
            S20b = S20f.imag
            S22a = S22f.real
            S22b = S22f.imag
            S21a = S21f.real
            S21b = S21f.imag

            if real:
                C20b *= -1.0
                C22b *= -1.0
                C21b *= -1.0
                S20b *= -1.0
                S22b *= -1.0
                S21b *= -1.0

            if test:
                colat = np.radians(40.0)
                lon = np.radians(9.0)
                if i > NP:
                    U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t_*P, colat, lon)
                elif i < NP:
                    U = potential.potential_moonmoon(a2, a1, m2, R1, n2, n1, n2, t_*P, colat, lon)

                # plt.plot(freqs, abs(S22f)**2.0)
                # plt.show()
                Utest = U.copy()
                Utest2 = U.copy()
                for x in range(len(t_)):
                    Utest[x] = MakeGridPoint(clmt[x], np.degrees(np.pi/2. - colat), np.degrees(lon), norm=3, csphase=-1)

                Plm = PLegendreA(3, np.cos(colat))#PlmBar(3, np.cos(colat))
                if real:
                    q = freqs[1:FM]
                    tq, q = np.meshgrid(t_, q)
                    tt, C20a_t = np.meshgrid(t_, C20a)
                    tt, C20b_t = np.meshgrid(t_, C20b)
                    tt, C22a_t = np.meshgrid(t_, C22a)
                    tt, C22b_t = np.meshgrid(t_, C22b)
                    tt, S22a_t = np.meshgrid(t_, S22a)
                    tt, S22b_t = np.meshgrid(t_, S22b)

                    C20_T = 0.5 * C20a[0] + np.sum(C20a_t[1:]*np.cos( q*(n1-n2)*tq*P) + C20b_t[1:] * np.sin( q*(n1-n2)*tq*P), axis=0)
                    C22_T = 0.5 * C22a[0] + np.sum(C22a_t[1:]*np.cos( q*(n1-n2)*tq*P) + C22b_t[1:] * np.sin( q*(n1-n2)*tq*P), axis=0)
                    S22_T = 0.5 * S22a[0] + np.sum(S22a_t[1:]*np.cos( q*(n1-n2)*tq*P) + S22b_t[1:] * np.sin( q*(n1-n2)*tq*P), axis=0)
                    Utest2 = Plm[3] * (C20_T * np.cos(0*lon)) + Plm[5] * (C22_T*np.cos(2*lon) + S22_T*np.sin(2*lon))

                else:
                    c20_c = np.zeros(FM, dtype=np.complex)
                    c20_c[:].real = C20a[:]
                    c20_c[:].imag = C20b[:]

                    c22_c = np.zeros(FM, dtype=np.complex)
                    c22_c[:].real = C22a[:]
                    c22_c[:].imag = C22b[:]

                    s22_c = np.zeros(FM, dtype=np.complex)
                    s22_c[:].real = S22a[:]
                    s22_c[:].imag = S22b[:]

                    tt, c20_ct = np.meshgrid(t_, c20_c)
                    tt, c22_ct = np.meshgrid(t_, c22_c)
                    tt, s22_ct = np.meshgrid(t_, s22_c)

                    iqnt = np.zeros(FM, dtype=np.complex)
                    q = freqs[:]

                    iqnt[:].imag =  q * abs(n1-n2)
                    tt, iqnt_ = np.meshgrid(t_, iqnt)
                    tq, q = np.meshgrid(t_, q)
                    C20_T = np.sum(c20_ct * np.exp(iqnt_*tq*P), axis=0)/2
                    C22_T = np.sum(c22_ct * np.exp(iqnt_*tq*P), axis=0)/2
                    S22_T = np.sum(s22_ct * np.exp(iqnt_*tq*P), axis=0)/2

                    Utest2 = Plm[3] * (C20_T * np.cos(0*lon)) + Plm[5] * (C22_T*np.cos(2*lon) + S22_T*np.sin(2*lon))
                    Utest2 = (Utest2.real + Utest2.imag)

                plt.plot(t_, U)
                plt.plot(t_,Utest2)
                # plt.plot(t_, Utest2/U)
                ax = plt.gca()
                ax.set_title(planets[P_ID]+"--"+planets[i])

                print(U/Utest2)
                plt.show()

            data_arr = np.array([C20a, C20b, C21a, C21b, C22a, C22b, S20a, S20b, S21a, S21b, S22a, S22b])

            # Save coefficients to the subgroup
            subgrp.create_dataset("tide coeffs", data=data_arr.T)
            subgrp.create_dataset("frequency", data=freqs, dtype=np.int)


if __name__ == '__main__':
    main()
