import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.misc import factorial
from pyshtools.expand import MakeGridDH
# from pyshtools.spharm_lm_ import spharm
import pyshtools
import string
from matplotlib import font_manager

from matplotlib.ticker import MultipleLocator, LogLocator

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\usepackage{libertinust1math}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

# plt.rc('pdf.fonttype', fonttype=42)
mpl.rcParams['pdf.fonttype'] = 42

def membraneUpsBeta(shell_thickness, ocean_thickness, freq, mass, radius, grav_surf, den_ocean, den_shell, n_max=12):
    def membraneFluidLoveNumbers(R, Rm, g, den_ocean, den_bulk, rigid):
        ''' Implementation of fluid love numbers - Eqs 57 and 58 from Beuthe (2015) - Tides on Europa'''

        # Define constants
        y = Rm/R 
        den_ratio = den_ocean/den_bulk 
        rigid_eff = rigid/(den_bulk * g * R)

        # Define polynomials from the propagator matrix (Eqs. 119, 120)

        A = 5./19. * (1 - den_ratio) * ( 2 + 3*y**5 - (2 - 5*y**3 + 3*y**5)*den_ratio )

        B = 10 - (16 - 25*y**3 + 9*y**5)*den_ratio + 3*(2 - 5*y**3 + 3*y**5)*den_ratio**2
        B *= 1./19. * (1 - den_ratio)

        h2_fluid = (A + 5*y**4*rigid_eff) / (B + (5 - 3*den_ratio) * y**4 * rigid_eff)
        k2_fluid = h2_fluid - 1.0 

        return k2_fluid, h2_fluid


    # d = np.loadtxt("love_numbers/europa_fluid_love_numbers.dat").T

    # k = d[1, :n_max]
    # h = d[2, :n_max]
    
    n = np.arange(1, n_max+1, 1)

    # sys.exit()


    G = 6.674e-11

    pois_ratio = 0.35
    rigid_shell = 3.5e9
    # rigid_shell = 60e9
    # rigid_shell  = (3.44 + 0.0001j) * 1e9

    visc_factor = rigid_shell/(freq*1e19)
    rigid_shell = (rigid_shell / ( 1 -  visc_factor*1j) )

    pois_ratio = (3*pois_ratio  - (1+pois_ratio)*visc_factor*1j) / (3 - 2*(1 + pois_ratio)*visc_factor*1j)

    # print(rigid_shell/1e9, pois_ratio)
    rigid_core = 40e9
    # pois_ratio = 0.5 
    # rigid_shell = 60e9

    # rigid_shell = (3.44 + 0.13j)*1e9
    # pois_ratio = 0.333 - 0.004j

    # rigid_shell = 3.5e9
    # pois_ratio = 0.33

    radius_core = radius #- ocean_thickness#(shell_thickness + ocean_thickness)
    radius_ocean = radius - (shell_thickness)

    beta = np.zeros(len(n), dtype=np.complex)
    ups = np.zeros(len(n), dtype=np.complex)


    mass_total = mass

    den_ocean = 1000.0
    den_shell = 1000.0

    vol_total = 4./3. * np.pi * radius**3.0


    den_bulk = mass_total/vol_total

    den_ratio= den_ocean/den_bulk


    spring = np.zeros(len(n), dtype=np.complex)
    for l in n:
        rigid_eff = (2*l*l + 4*l + 3)/l * rigid_core / (den_bulk * grav_surf * radius)

        rigid_factor = 1. / (1. + rigid_eff)

        # // FIND LOADING AND TIDAL LOVE NUMBERS
        
        if l==1:
            kt = 0.0 
            ht = 0.0
        else:
            ht = rigid_factor * (2*l + 1)/( ( 2 * (l-1) ) )
            kt = rigid_factor * 3./( ( 2 * (l-1) ) )

        kl = -rigid_factor
        hl = -rigid_factor * (2*l + 1) / 3.0

        gam_tide = 1.0 + kt - ht
        gam_load = 1.0 + kl - hl

        # // FIND THE SPRING CONSTANTS
        x = ((l - 1)*(l + 2))

        bendRigidity = rigid_shell * shell_thickness**3.0 \
                       / (6. * (1. - pois_ratio))

        sprMembrane = 2.*x * (1. + pois_ratio)/(x + 1. + pois_ratio) \
                      * rigid_shell / (den_ocean * grav_surf * radius) \
                      * shell_thickness / radius

        sprBending = x**2.0 * (x + 2.)/(x + 1. + pois_ratio) \
                     * bendRigidity / (den_ocean * grav_surf * radius**4.0)

        sprConst = sprMembrane + sprBending

        # // FIND dSpring and dGam^tide
        xi = 3.0 / (2.*l + 1.0) * (den_ratio) #// CHANGED den_core to den bulk and den_ocean to den_shell

        



        dsprConst = 1. - (1. + xi*hl)**2.0/(1. + xi*(ht - hl)*sprConst)
        dsprConst *= -sprConst

        dgam_tide = (1. + xi*hl)*ht / (1. + xi*(ht - hl)*sprConst)
        dgam_tide *= -sprConst

        # // Find beta and nu coefficients
        beta[l-1] = 1. - xi*gam_load + sprConst + dsprConst
        ups[l-1] = gam_tide + dgam_tide

        # Assume infinitely rigid mantle
        beta[l-1] = 1. - xi + sprConst 
        ups[l-1] = 1.0
        spring[l-1] = sprConst


    return beta, ups, spring

# plt.rcParams['figure.figsize'] = (7,4)

labelsize=9
ticksize=8

p_coeffs = h5py.File("galilean_coefficients.h5", 'r')
ecc_coeff = p_coeffs['europa']['eccentricity']['tide coeffs'][:]
io_coeff = p_coeffs['europa']['io']['tide coeffs'][:]

p_coeffs = (ecc_coeff + io_coeff)*0.5

p_coeffs = np.concatenate( (p_coeffs[40:100], p_coeffs[101:-39]), axis=0 )
# p_coeffs = np.concatenate( (p_coeffs[98:100], p_coeffs[101:101+3]), axis=0 )

# p_coeffs = p_coeffs[:-40]

# print(p_coeffs[:,0])
U20 = p_coeffs[:, 0]
U22_C = p_coeffs[:, 4]
U22_S = p_coeffs[:, 11]

love_file = h5py.File("shell_love_numbers.h5", 'r')
beta = love_file['europa']["beta"][10 - 1][1]
ups = love_file['europa']["upsilon"][10 - 1][1]

g = 1.3079460990117284

p_coeffs *= ups/beta * 1./g

r = 1560.8e3
force_freq = 2.05e-5*0.5

# hsim = 73.2186e3 - 0.1
hsim = 73.2181e3

hres = 73218.06615773
hsim = hres
hsim = 71e3
# hsim = 4e3
# hsim =73.0e3

target_h = [80e3, 53e3]

target_h = [68103.27196342]

file_names = ["/home/hamish/Research/LaplaceTidalAnalytic/workflow/res_europa_68km.h5"]
# file_names = ["/home/hamish/Research/LaplaceTidalAnalytic/workflow/m2_test.h5"]




labels = ['80 km', '50 km']
lines = []
for x in range(1):
    infile = h5py.File(file_names[x], 'r')
    # infile2 = h5py.File(file_names[x], 'r')
    # infile = h5py.File("/home/hamish/Research/LaplaceTidalAnalytic/workflow/testin.h5", 'r')

    ho = infile['europa']['ocean thickness'][:]
    freqs = infile['europa']['frequency'][:]

    idx = (np.abs(ho - target_h[x])).argmin()
    # idx += 1

    phi = infile['europa']['velocity potential'][idx]

    # for i in range(len(freqs)):
    #     print(abs(phi[i,2]), freqs[i]) 

    n_max = len(phi[0, 0, :])
    n = np.arange(1, n_max+1, 1)

    ff, nn = np.meshgrid(freqs, n)

    Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
    Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
    Nnm2[0] = 1.0

    eta_freq = np.zeros(len(freqs))
    eta_freq_eq = np.zeros(len(freqs))
    eta_max = 0

    for i in range(len(eta_freq)):

        beta, ups, spring = membraneUpsBeta(10e3, 50e3, abs(force_freq*freqs[i]), 1.0, r, g, 1e3, 1e3, n_max=4)


        cilm = np.zeros((2, n_max+1, n_max+1))
        eta_eq_cilm = np.zeros((2, n_max+1, n_max+1))

        # phi_m2 = np.nansum( phi[i,2,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i])), axis=0)
        # phi_m0 = np.nansum( phi[i,0,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i])), axis=0)
        phi_m2 = phi[i,2,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i]))
        phi_m0 = phi[i,0,:] * n*(n+1)/r**2.0 * (ho[idx]/(-(force_freq)*freqs[i]))

        cilm[0,1:,2] = -phi_m2.imag
        cilm[1,1:,2] =  phi_m2.real
        cilm[0,1:,0] = -phi_m0.imag
        cilm[1,1:,0] =  phi_m0.real

        

        # print(freqs[i], U20[i], U22_C[i], U22_S[i])

        eta_eq_cilm[0, 2, 2] = U22_C[i] + U22_S[i]
        eta_eq_cilm[0, 2, 0] = U20[i]

        # print(cilm)

        griddh = MakeGridDH(cilm, norm=3, sampling=2)
        if i == 0:
            displacement = griddh.copy()
            disp_coeff = cilm.copy()

        else:
            displacement += griddh
            disp_coeff += cilm

        eta_freq[i] = np.amax(np.amax(abs(griddh)))

        # print(np.shape(griddh))

        # eta_freq[i] = abs(griddh[13,0])
        # print(freqs[i], abs(griddh[13,0]))
        # eta_max += griddh[4,4]

        griddh_eq = MakeGridDH(eta_eq_cilm, norm=3, sampling=2)

    plt.contourf(displacement, 11)
    plt.colorbar()

    fig = plt.gcf()

    fig.savefig("/home/hamish/Dropbox/Tests/m2_tide.png", dpi=200)
    plt.show()

        # eta_freq_eq[i] = abs(griddh_eq[13,0])

    # l_eq, = axes[x].step(freqs/2+0.25, eta_freq_eq, 'k-', lw=0.4)

    # l_h = axes[x].fill_between(freqs/2+0.25, eta_freq, facecolor=colors[x], step="pre",  alpha=0.8)

    # ax3.fill_between(freqs/2+0.25, eta_freq/eta_freq_eq, facecolor=colors[x], step="pre", alpha=0.8)



    # if x is 0:
    #     lines.append(l_eq)

    # lines.append(l_h)


# axes[0].legend(lines, ['Equilibrium tide', 'Dynamic tide, $h_o=$ \\SI{80}{\\km}', 'Dynamic tide, $h_o=$ \\SI{50}{\\km}'], frameon=False, prop={'size': 6}, loc='upper left')

# for ax in axes:
#     ax.set_yscale('log')
#     ax.set_ylim([1e-6, 35])
#     ax.set_xlim([-30, 30])
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)

#     # ax.xaxis.set_tick_params(which='minor', top = True, width=0.5, direction='out', zorder=100000)

# for ax in axes[:2]:
#     ax.axes.xaxis.set_ticklabels([])
#     ax.set_ylabel("Tidal\namplitude [m]", fontsize=labelsize)

#     locmin = LogLocator(base=10.0, subs=(1.0, ), numticks=100)
#     ax.yaxis.set_major_locator(locmin)
#     labels = ["\\num[retain-unity-mantissa = false]{{{:1.0e}}}".format(item) for item in ax.get_yticks().tolist()]
#     labels[::2] = [''] * len(labels[::2])
#     ax.set_yticklabels(labels, fontsize=ticksize)

#     # ax.yaxis.set_minor_locator(LogLocator(base=10,numticks=12))
#     # ax.yaxis.set_tick_params(which='minor', direction='out', width=0.5, length=1.5)
#     # ax.xaxis.set_tick_params(which='minor', bottom = False)
#     # ax.axes.yaxis.set_ticklabels([])

# ax3.set_ylim([1.0, 300])
# ax3.set_xlabel("Tidal frequency, $q$", fontsize=labelsize)
# ax3.set_ylabel("Dynamic tide\nperturbation, $\\eta/\\eta_{eq}$", fontsize=labelsize)

# ax3.xaxis.set_minor_locator(MultipleLocator(1))
# ax3.xaxis.set_tick_params(which='minor', direction='out', top=False, width=0.5, length=1.5)
# # locmin = LogLocator(base=10,numticks=12)
# # ax3.yaxis.set_minor_locator(locmin)
# ax3.yaxis.set_tick_params(which='minor', direction='out', width=0.5, length=1.5)

# labels = ["{:d}".format(int(item)) for item in ax3.get_xticks().tolist()]
# ax3.set_xticklabels(labels, fontsize=ticksize)
# labels = ["{:d}".format(int(item)) for item in ax3.get_yticks().tolist()]
# ax3.set_yticklabels(labels, fontsize=ticksize)


# ax4 = fig.add_subplot(gs[:,1])

# N = 2
# lons = np.arange(0, 360., N, dtype=float)
# colats = np.arange(0.0, 180., N, dtype=float)
# # fig, ax = plt.subplots(figsize=(4,8))

# data = np.loadtxt("europa_73km_res.txt")
# data_io = np.loadtxt("io_80km_res.txt")

# data_merid = np.mean(data, axis=1)
# data_merid_io = np.mean(data_io, axis=1)

# p = ax4.plot(data_merid, 90-colats, lw=1.8, label = 'Europa, $h_0 =$ \\SI{73.218}{km}')
# p = ax4.plot(data_merid_io, 90-colats, lw=1.8, label = 'Io, $h_0 =$ \\SI{80.245}{km}')
# ax4.set_ylim([-90,90])

# ax4.xaxis.set_major_locator(MultipleLocator(0.25))
# ax4.xaxis.set_tick_params(which='minor', direction='out', top=False, width=0.5, length=1.5)

# labels = ["{:.2f}".format(item) for item in ax4.get_xticks().tolist()]
# ax4.set_xticklabels(labels, fontsize=ticksize)
# labels = ["{:d}".format(int(item)) for item in ax4.get_yticks().tolist()]
# ax4.set_yticklabels(labels, fontsize=ticksize)

# ax4.set_ylabel("Latitude [\\si{\\degree}]", fontsize=labelsize)
# ax4.set_xlabel("Zonally-averaged heat flux [\\si{\\watt\\per\\metre\\squared}]", fontsize=labelsize)

# ax4.legend(frameon=False, prop={'size': 6}, loc='upper right')

# font = font_manager.FontProperties(family='STIXGeneral',
#                                    weight='bold',
#                                    style='normal', size=12)

# tt = ax1.text(-0.12, 1.0, "\\textbf{a}", fontfamily="sans-serif", transform=ax1.transAxes,
#             size=11, weight='bold')

# tt = ax4.text(1.1, 1.0, "\\textbf{b}", fontfamily="sans-serif", transform=ax1.transAxes,
#             size=11, weight='bold')


# # plt.setp(tt.texts, family='Consolas')

# fig.savefig("/home/hamish/Dropbox/Tests/disp_europa.pdf", bbox_inches='tight', dpi=1000)

# disp_coeff = pyshtools.expand.SHExpandDH(displacement, norm=3, sampling=2)
#
# fig, ax = plt.subplots()
#
# power_per_l = pyshtools.spectralanalysis.spectrum(disp_coeff, normalization='unnorm')
# degrees = np.arange(disp_coeff.shape[1])
#
# # cnt = ax.semilogy(degrees[:8], power_per_l[:8], 'o')
# # ax.set_ylim([1e-8, 1e5])
#
# cnt = ax.pcolormesh(displacement)
# plt.colorbar(cnt)
#
# fig.savefig("/home/hamish/Dropbox/Tests/disp_europa2.pdf", bbox_inches='tight')

plt.show()
