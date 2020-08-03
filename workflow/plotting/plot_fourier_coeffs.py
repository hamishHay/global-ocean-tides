
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

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Avante Garde'
plt.rcParams['mathtext.it'] = 'Avante Garde:italic'
plt.rcParams['mathtext.bf'] = 'Avante Garde:bold'

# plt.rc('font',sans_serif="Avante Garde")
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avante Garde']})
# plt.rc('font',serif='Palatino')
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx},\\usepackage{sfmath}')

plt.rc('axes', linewidth=1.0)

plt.rc('figure', dpi=400)
plt.rcParams['pdf.fonttype'] = 42


database = Database().moon_dict

# grid spacing in latitiude and longitude for SH computation
dx = 6.0
dy = 6.0

# max spherical harmonic degree for calculation
LMAX = 2

# time spacing
dt = 0.001
FM = 200

DAY = 24*60**2

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
    save_file = h5py.File('enceladus_coefficients_real.h5','w')
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

    planets = ['europa']#, 'ganymede', 'callisto']
    # planets = ['europa']
    forcings = [['io', 'ganymede']]#, 'eccentricity']]
                # ['europa', 'callisto', 'eccentricity'],
                # ['ganymede', 'eccentricity']]

    # forcings = [['eccentricity']]
    eur_gan = database['Europa'].mean_motion - database['Ganymede'].mean_motion
    gan_cal = database['Callisto'].mean_motion - database['Ganymede'].mean_motion
    # eur_gan = database['Europa'].mean_motion
    nij = np.array([[eur_gan, eur_gan, 0], [eur_gan, eur_gan, eur_gan], [eur_gan, gan_cal, eur_gan], [gan_cal, database['Callisto'].mean_motion, 0] ])
         # database['Europa'].mean_motion - database['Ganymede'].mean_motion,
         # database['Europa'].mean_motion - database['Ganymede'].mean_motion,
         # ]

    # planets = ['enceladus']
    # forcings = [['eccentricity']]
    nij = database['Europa'].mean_motion

    col = ['C0', 'C1', 'C3']
    phase = [0.5, 0.0]
    P = abs(2*np.pi/nij)

    NP = len(planets)  # Number of planets

    real=False
    test=False

    FM=2000

    # Create 1D arrays in colatitue, longitude, and time
    colats_ = np.radians(np.arange(0., 180., dy, dtype=np.float64))
    lons_   = np.radians(np.arange(0., 360., dx, dtype=np.float64))
    t_      = np.linspace(-1.0, 1.0, FM, endpoint=False, dtype=np.float64)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 4))
    coeffs = np.zeros(FM)

    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(10, 2.6))
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
        o1 = np.radians(database[planet.capitalize()].obliquity)


        # Create group for tides raised on planet j
        grp = save_file.create_group(planet)

        header = '  '



        # P = 2*abs(np.pi/(n1 - n2))
        # Loop through each tide raising planet
        for i in range(len(forcings[j])):
            forcing = forcings[j][i]
            # Create subgroup for the tide raising planet i

            subgrp = grp.create_group(forcing)

            if forcing is not 'eccentricity' and forcing is not 'obliquity':
                # Get planet i data
                a2 = database[forcing.capitalize()].semimajor_axis
                m2 = database[forcing.capitalize()].mass
                n2 = database[forcing.capitalize()].mean_motion

                # Get conjuction/forcing period
                # P = 2*abs(np.pi/(n1 - n2))

                # Create 3D grid in longitude, colatitude, and time
                lons, colats, t = np.meshgrid(lons_, colats_, (t_+phase[i])*P)

                # Reverse Semimajor axes if tide raiser is inside or outside
                U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t, colats, lons)

                # if a2 > a1:
                #     U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n1, t, colats, lons)
                # else:
                #     U = potential.potential_moonmoon(a1, a2, m2, R1, n1, n2, n2, t, colats, lons)


            else:
                # P = 2*abs(np.pi/n1)
                lons, colats, t = np.meshgrid(lons_, colats_, t_*P)
                if forcing is "eccentricity":
                    # U = potential.potential_ecc_lib_east(e1, R1, n1, t, colats, lons)
                    # M = Database().saturn.mass
                    M = database[planet.capitalize()].parent.mass

                    # e1 = 0.0000000000001
                    # o1 = np.radians(0.01)
                    U = potential.potential_kaula(2, 0, 0, M, a1, R1, n1, e1, o1, t, colats, lons)
                    # U += potential.potential_kaula(2, 1, 0, M, a1, R1, n1, e1, o1, t, colats, lons+np.pi/2.)
                    U += potential.potential_kaula(2, 2, 0, M, a1, R1, n1, e1, o1, t, colats, lons)
                    for q in range(1, 6+1):
                        U += potential.potential_kaula(2, 0, q, M, a1, R1, n1, e1, o1, t, colats, lons)
                        # U += potential.potential_kaula(2, 1, q, M, a1, R1, n1, e1, o1, t, colats, lons+np.pi/2.0)
                        U += potential.potential_kaula(2, 2, q, M, a1, R1, n1, e1, o1, t, colats, lons)

                    # print(U)

                else:

                    U = potential.potential_ecc(e1, R1, n1, t, colats, lons)

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

            ax1.plot(t_*P/DAY, C20)
            ax2.plot(t_*P/DAY, C22)
            ax3.plot(t_*P/DAY, S22)

        ax1.set_xlabel("Time [Days]")
        ax2.set_xlabel("Time [Days]")
        ax3.set_xlabel("Time [Days]")

        ax1.set_title("$C_{20}$")
        ax2.set_title("$C_{22}$")
        ax3.set_title("$S_{22}$")

        ax1.set_xlim([-P/DAY, P/DAY])
        ax2.set_xlim([-P/DAY, P/DAY])
        ax3.set_xlim([-P/DAY, P/DAY])
        
        ax1.set_ylabel("Tidal potential\ncoefficient [\si{\metre\squared\per\second\squared}]")
        # ax2.set_xlabel("Time [Days]")
        # ax3.set_xlabel("Time [Days]")

        ax2.annotate('Io',
            xy=(-2, 0.16), xycoords='data',
            xytext=(-3, 0.2), textcoords='data', color='C0',
            arrowprops=dict(edgecolor='C0', arrowstyle='-'))

        ax2.annotate('Ganymede',
            xy=(0, 0.10), xycoords='data',
            xytext=(-0.6, 0.13), textcoords='data', color='C1',
            arrowprops=dict(edgecolor='C1', arrowstyle='-'))
            # horizontalalignment='right', verticalalignment='top')

        ax1.set_xticks(np.arange(-3, 3.1, 1.0))
        ax2.set_xticks(np.arange(-3, 3.1, 1.0))
        ax3.set_xticks(np.arange(-3, 3.1, 1.0))

        fig.savefig("/home/hamish/Dropbox/Tests/tide_coeffs_europa.pdf", bbox_inches='tight')

        plt.show()

if __name__ == '__main__':
    main()
