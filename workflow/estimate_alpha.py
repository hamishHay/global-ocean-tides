import numpy as np 
import h5py
from scipy.misc import factorial 
import matplotlib.pyplot as plt

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

moon_name = 'europa'
forcing_name = ['eccentricity', 'io']





ho = 20e3 
radius = 1565e3 
g = 1.31
rot_rate = 2.05e-5
beta = 1.0
cd = 0.002
ecc = 0.0094
uo = rot_rate**3.0 * radius**3 / (g * ho)

def u2a(u):
    return cd/ho*u

n = 2
Nn2m2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)


n = 3
Nn3m2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
axes = [ax1, ax2]

for i in range(2):
    coeff_file = h5py.File("galilean_coefficients.h5", 'r')
    # coeff_file = h5py.File("enceladus_coefficients_real.h5", 'r')
    # coeff_file = h5py.File("galilean_coefficients.h5", 'r')
    freqs  = coeff_file[moon_name][forcing_name[i]]['frequency q'][:]
    q_nij  = coeff_file[moon_name][forcing_name[i]]['frequency'][:]


    coeffs = coeff_file[moon_name][forcing_name[i]]['tide coeffs'][:, :] * 0.5

    vels = np.zeros(len(freqs))
    alphas = np.zeros(len(freqs))
    for j in range(len(freqs)):

        a_20c = coeffs[j][0]
        a_22c = coeffs[j][4]
        a_21c = coeffs[j][2]
        b_21s = coeffs[j][9]
        b_22s = coeffs[j][11]

        forcing_freq = -q_nij[j]

        lam = forcing_freq / (2 * rot_rate)
        eps = 4 * rot_rate**2.0 * radius**2.0 / (g * ho)

        forcing = a_22c + b_22s

        
        # --------------------------------------------------------------
        # Solve for libration component, P22
        if abs(a_22c + b_22s) > 1e-11 and forcing_freq != 0.0:
            q2 = 2/15. 
            p3 = 20./21.
            L3 = lam + 1./6.
            K2 = lam + 1./3. - beta*6/ (eps * lam)

            PHI = 1/(2*rot_rate) * forcing * ( (-1j*K2) -p3*q2 / (1j*L3))**(-1)

            PSI = -q2 / (1j*L3) * PHI

            # print(abs(PHI), abs(PSI))


            Ekin = np.pi * 1000 * ho * (Nn2m2 * abs(PHI)**2.0 + Nn3m2*abs(PSI)**2.0)
            vel = np.sqrt(Ekin / (4 * np.pi * radius**2.0 *ho * 1000 * 0.5))
            alpha = cd / ho * vel
            if freqs[j] > 0:
                vel_scal = np.sqrt(3/224.) * uo * ecc
            else:
                vel_scal = np.sqrt(63/160.) * uo * ecc

            PHI_MAG = abs(forcing) * eps / (24 * rot_rate)
            PSI_MAG = 2 * np.sqrt(1/7.) * PHI_MAG

            print(freqs[j]/2, forcing_freq, vel, vel_scal, alpha)

            vels[j] = vel
            alphas[j] = alpha

    freqs = np.array(list(freqs), dtype=np.float)
    # print(freqs)
    freqs /= 2
    axes[i].semilogy(-freqs, vels, 'o', label="Freq-dependent scaling")


    ax12 = axes[i].twinx()

    if i ==0:
        ymax = np.array(axes[i].get_ylim())
        ymax2 = u2a(ymax)

        ax12.set_ylim(np.log10(ymax2))
        ax12.set_ylabel("Linear drag coefficient, $\log \left(\\alpha \, [\si{\per\second}] \\right)$")
    else:
        ymax = np.array(axes[0].get_ylim())
        ymax2 = u2a(ymax)

        axes[i].set_ylim(ymax)
        ax12.set_ylim(np.log10(ymax2))
        ax12.set_ylabel("Linear drag coefficient, $\log \left(\\alpha \, [\si{\per\second}] \\right)$")
    # ax12.set_yscale('log')

    # secax = ax1.secondary_yaxis('right', functions=(u2a))

    # ax2.semilogy(-freqs, alphas, 'o')

    if i == 0:
        ax1.semilogy([-1, 1], [np.sqrt(3/224.) * uo * ecc, np.sqrt(63/160.) * uo * ecc], 'x', label="Chen et al (2014)")
    # ax1.semilogy([2], , 'x')

    if i == 0:
        axes[i].set_xlim([-4, 4])
    else:
        axes[i].set_xlim([-20, 20])
    # ax2.set_xlim([-10, 10])

    axes[i].set_ylabel("Mean flow speed [\si{\metre\per\second}]")


    



# ax2.set_ylabel("Linear drag coefficient, $\\alpha$ [\si{\per\second}]")

ax1.legend()

a = 1 
b = 1/3. * rot_rate
c = -6 * g * ho/ radius**2.0
res_freq_east = (-b + np.sqrt(b**2.0 - 4 * a * c))/ (2*a) 
res_freq_west = (-b - np.sqrt(b**2.0 - 4 * a * c))/ (2*a) 

# ax2.axvline(res_freq_east/rot_rate)
# ax2.axvline(res_freq_west/rot_rate)


ax1.set_title("Eccentricity-forcing, $h=\SI{20}{\km}$")
ax2.set_title("Io-forcing, $h=\SI{20}{\km}$")

ax1.set_xlabel("Frequency, $q$")
ax2.set_xlabel("Frequency, $q$")

plt.subplots_adjust(wspace=0.6)

fig.savefig("/home/hamish/Dropbox/Tests/alpha_estimate.pdf", bbox_inches='tight')

