
import sys
sys.path.append("../")

# from LTE_solver import LTESolver

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
dx = 15.0
dy = 15.0

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
    save_file = h5py.File('europa_io_a_coeffs_small.h5','w')
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
                ['tehtys']]

    planets = ['io', 'europa', 'ganymede', 'callisto']
    # planets = ['europa']
    forcings = [['europa', 'eccentricity'],
                ['io', 'ganymede', 'eccentricity'],
                ['europa', 'callisto', 'eccentricity'],
                ['ganymede', 'eccentricity']]

    # forcings = [['eccentricity']]
    eur_gan = database['Europa'].mean_motion - database['Ganymede'].mean_motion
    gan_cal = database['Callisto'].mean_motion - database['Ganymede'].mean_motion
    # eur_gan = database['Europa'].mean_motion
    nij = np.array([[eur_gan, eur_gan, 0], [eur_gan, eur_gan, eur_gan], [eur_gan, gan_cal, eur_gan], [gan_cal, database['Callisto'].mean_motion, 0] ])
         # database['Europa'].mean_motion - database['Ganymede'].mean_motion,
         # database['Europa'].mean_motion - database['Ganymede'].mean_motion,
         # ]

    # planets = ['europa']
    # forcings = [['io']]

    planets = ['europa']
    forcings = [['io']]#, 'eccentricity']]
    # planets = ['dione']
    # forcings = [['tethys']]
    # nij = np.array([[database['Enceladus'].mean_motion]])

    col = ['C0', 'C1', 'C3']
    phase = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    

    NP = len(planets)  # Number of planets

    real=False
    test=False

    FM=200

    # Create 1D arrays in colatitue, longitude, and time
    colats_ = np.radians(np.arange(0., 180., dy, dtype=np.float64))
    lons_   = np.radians(np.arange(0., 360., dx, dtype=np.float64))
    t_      = np.linspace(0., 1.0, FM, endpoint=False, dtype=np.float64)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 4))
    coeffs = np.zeros(FM)
    # Loop through each planet
    planet = planets[0]

    a1_moon = database[planet.capitalize()].semimajor_axis
    R1 = database[planet.capitalize()].radius
    n1 = database[planet.capitalize()].mean_motion
    e1 = database[planet.capitalize()].eccentricity
    o1 = np.radians(database[planet.capitalize()].obliquity)

    P = abs(2*np.pi/n1)

    
    

    semimajoraxes = np.linspace(0.99, 1.01, 801)
    COEFFS = np.zeros((len(semimajoraxes), FM, 12))
    FREQS = np.zeros((len(semimajoraxes), FM))
    SMA = np.zeros((len(semimajoraxes)))
    SMA = a1_moon*semimajoraxes

    grp = save_file.create_group(planets[0])
    subgrp = grp.create_group(forcings[0][0])

    lons, colats, t = np.meshgrid(lons_, colats_, t_)
    for j in range(len(semimajoraxes)):
        
        # P_ID = j

        # Get planet P data:
        # a1   = planet_data.p_a[0]
        # R1   = planet_data.p_R[0]
        # n1   = planet_data.p_n[0]
        # e1   = planet_data.p_e[0]
        # o1   = planet_data.p_o[0]
        a1 = a1_moon * semimajoraxes[j]
        n1 = np.sqrt(database[planet.capitalize()].parent.mass*G/a1**3.0)


        # Create group for tides raised on planet j
        

        header = '  '

        print(j/len(semimajoraxes))

        # P = 2*abs(np.pi/(n1 - n2))
        # Loop through each tide raising planet
        for i in range(len(forcings[0])):
            forcing = forcings[0][i]
            # Create subgroup for the tide raising planet i

            

            if forcing is not 'eccentricity' and forcing is not 'obliquity':
                # Get planet i data
                a2 = database[forcing.capitalize()].semimajor_axis #a1*semimajoraxes[j]
                m2 = database[forcing.capitalize()].mass
                n2_old = database[forcing.capitalize()].mean_motion

                n2 = 2*n1 
                a2 = (database[planet.capitalize()].parent.mass*G/n2**2.0)**(1./3.)

                # n2 = np.sqrt(database[planet.capitalize()].parent.mass*G/a2**3.0)
                


                # Get conjuction/forcing period
                P = 2*abs(np.pi/(n1 - n2))
                
# 
                # Create 3D grid in longitude, colatitude, and time
               

                # Reverse Semimajor axes if tide raiser is inside or outside
                U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t*P, colats, lons)

                # if a2 > a1:
                #     U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t*P, colats, lons)
                # else:
                #     U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n2, t*P, colats, lons)


            # else:
                # P = 2*abs(np.pi/n1)
                # lons, colats, t = np.meshgrid(lons_, colats_, t_*P)
                # if forcing is "eccentricity":
                    # U = potential.potential_ecc_lib_east(e1, R1, n1, t*P, colats, lons)
                    # M = Database().saturn.mass
                # M = database[planet.capitalize()].parent.mass
                # n2 = 0

                # # e1 = 0.0000000000001
                # # o1 = np.radians(0.01)
                # U += potential.potential_kaula(2, 0, 0, M, a1, R1, n1, e1, o1, t*P, colats, lons)
                # # U += potential.potential_kaula(2, 1, 0, M, a1, R1, n1, e1, o1, t*P, colats, lons+np.pi/2.)
                # U += potential.potential_kaula(2, 2, 0, M, a1, R1, n1, e1, o1, t*P, colats, lons)
                # for q in range(1, 6+1):
                #     U += potential.potential_kaula(2, 0, q, M, a1, R1, n1, e1, o1, t*P, colats, lons)
                #     # U += potential.potential_kaula(2, 1, q, M, a1, R1, n1, e1, o1, t*P, colats, lons+np.pi/2.0)
                #     U += potential.potential_kaula(2, 2, q, M, a1, R1, n1, e1, o1, t*P, colats, lons)

                #     # print(U)

                # else:

                #     U = potential.potential_ecc(e1, R1, n1, t*P, colats, lons)

                # U = potential.potential_obl(o1, R1, n1, t, colats, lons)
            # plt.plot(t_, U[5,5])
            #
            # plt.show()

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

            # coeffs += C22f.real + S22f.imag



            if real:
                C20b *= -1.0
                C22b *= -1.0
                C21b *= -1.0
                S20b *= -1.0
                S22b *= -1.0
                S21b *= -1.0

            # if i==2:
            # ax3.semilogy(freqs, abs(S22b), 'o', markersize=4, c=col[i])
            # ax2.semilogy(freqs, abs(C22a), 'o', markersize=4, c=col[i])
            # ax1.semilogy(freqs, abs(C20a), 'o', markersize=4, c=col[i])
            # ax4.semilogy(freqs, abs(C22a + S22b), 'o', markersize=4, c=col[i])
            # ax4.plot(freqs, C21a + S21b, 'o', markersize=4, c=col[i])

            # if forcing is 'eccentricity':
            #     print(C20a[abs(C20a) > 1e-12])
            #     print(C22a[abs(C22a) > 1e-12])
            #     print(S22b[abs(S22b) > 1e-12])
            #
            #     print(np.mean(S22))

            # POWA[j,:] = np.sqrt((C22a + S22b)**2)
            # FREQ_MAG[j, :] = freqs*abs(n1-n2)
            

            # data_arr = np.array([C20a, C20b, C21a, C21b, C22a, C22b, S20a, S20b, S21a, S21b, S22a, S22b])
            nij = (n1 - n2)
            COEFFS[j] = np.array([C20a, C20b, C21a, C21b, C22a, C22b, S20a, S20b, S21a, S21b, S22a, S22b]).T
            FREQS[j] = freqs*nij

            # Save coefficients to the subgroup
            # subgrp.create_dataset("tide coeffs", data=data_arr.T)
            # subgrp.create_dataset("frequency q", data=freqs, dtype=np.int)
            # subgrp.create_dataset("frequency", data=freqs*nij[j, i], dtype=np.float)

        # ax.step(freqs, coeffs,where='mid')
        # axes = [ax1, ax2, ax3, ax4]
        # for ax in axes:
        #     ax.set_xlim([-20,20])
        #     # ax.set_ylim([1e-4,10])
        #     ax.set_xlabel("Frequency / [$n_E - n_G$]")
        #     ax.grid(which='both', alpha=0.2)

        # ax1.set_title("$\Re(A_{20q})$")
        # ax2.set_title("$\Re(A_{22q})$")
        # ax3.set_title("$\Im(B_{22q})$")
        # ax4.set_title("$\Re(A_{22q}) + \Im(B_{22q})$")
        #
        #
        # ax1.set_ylabel("Tidal potential")

    subgrp.create_dataset("tide coeffs", data=COEFFS)
    subgrp.create_dataset("sma", data=SMA)
    subgrp.create_dataset("frequency", data=FREQS)

    # fig, ax = plt.subplots()
    # ax.pcolormesh(semimajoraxes, freqs, np.log10(POWA).T)
    # ax.contour(semimajoraxes, freqs, np.log10(POWA).T, colors='k')
    # ax.set_xscale('log')
    # ax.set_ylim([-40,40])
    # ax.axvline(database[forcings[0][0].capitalize()].semimajor_axis/database[planets[0].capitalize()].semimajor_axis, linestyle='--')
    # fig.savefig("/home/hamish/Dropbox/Tests/A_B_coefficients_io_gan.pdf", bbox_inches=False)

    # fig, ax = plt.subplots()

    # for x in [0]:
    #     ax.scatter(abs(FREQ_MAG[x,101:])/database[planets[0].capitalize()].mean_motion, POWA[x,101:], c='b')
    #     ax.scatter(abs(FREQ_MAG[x,:100])/database[planets[0].capitalize()].mean_motion, POWA[x,:100], c='r')
    # ax.legend()
    # # ax.contour(semimajoraxes, freqs, np.log10(POWA).T, colors='k')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim([1e-10,1e2])
    # # ax.axvline(database['Io'].semimajor_axis/database['Europa'].semimajor_axis, linestyle='--')
    # ax.set_xlabel("forcing freq/rotation freq")
    # ax.set_ylabel("l=2, m=2, forcing magnitude")
    # ax.set_title("{:s} forcing power at {:s} for different semimajor axes".format(forcings[0][0], planets[0]))

    # fig.savefig("/home/hamish/Dropbox/Tests/{:s}_forcing_at_{:s}.pdf".format(forcings[0][0], planets[0]), bbox_inches=False)
    
    # plt.show()

if __name__ == '__main__':
    main()
