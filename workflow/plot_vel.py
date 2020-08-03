import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.misc import factorial
from scipy.interpolate import interp1d
from pyshtools.legendre import PLegendreA_d1
from pyshtools.expand import SHExpandDH
from pyshtools.expand import MakeGridDH
import planetPy
import matplotlib.font_manager as font_manager

# plt.rc('text', usetex=True)

# def get_

#
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Avante Garde']})
# fontpath = '/usr/share/fonts/opentype/linux-libertine/LinLibertine_DR.otf'
# prop = font_manager.FontProperties(fname=fontpath)
# plt.rcParams['font.family'] = prop.get_name()
# plt.rcParams['text.usetex'] = True

plt.rc('text', usetex=True)
plt.rc('text.latex',preamble='\\usepackage{siunitx}, \\usepackage{libertine}, \\sisetup{detect-all}')

plt.rc('lines', linewidth=0.6)

plt.set_cmap('bone')

alpha = 1e-11

den = 1e3
moon = 'europa'
h = 100e3

# den = 3e3
# moon = 'io'
# h = 80244.83145156

data = planetPy.database.Database()
moons = data.jupiter.moons
radius = moons[moon.capitalize()].radius

fig, ax = plt.subplots(figsize=(8,4))



# infile = h5py.File("LTE_solutions_alpha_" + str(alpha)+ ".h5", 'r')
# infile = h5py.File("callisto_benchmark", 'r')
infile = h5py.File("eur_res_h68.1km_v1e18_a11.h5", 'r')
# infile = h5py.File("negative_test.h5", 'r')

# infile = h5py.File("io_heat_80.h5", 'r')

# infile = h5py.File("m2_test.h5", 'r')



freqs = list(infile[moon]['frequency'][:])
# omegas = list(infile[moon]['frequency q'][:])

# print(omegas)

ho = infile[ moon ]['ocean thickness'][:]


phi = infile[moon]['velocity potential'][:]
psi = infile[moon]['stream function'][:]

# diss = infile[moon]['dissipated power'][:] #+ infile[moon]['dissipated power: shell'][:]

infile.close()


n_max = len(phi[0, 0, 0, :])
dt = 0.01


idx = (np.abs(ho - h)).argmin()

x = idx

# print(diss)
# print(np.sum(diss[:,x])/1e9)

nij = 4.357657208208119e-06#0.5*2.05e-5#5.892342791791881e-06#2.05e-5*0.5
P = 2*np.pi/(nij)
times = np.arange(0, 1.0, dt)
times = [0.0]
count = 0


for time in times:
    t=(time+0.0)*P 
    
    FLUX_DT = 0

    cilm_phi = np.zeros((2, n_max+1, n_max+1))
    cilm_psi = np.zeros((2, n_max+1, n_max+1))
    cilm_eta = np.zeros((2, n_max+1, n_max+1))
    
    ETA = None
    for m in [1]:
        for n in range(4):  
            for q in range(len(freqs)):
                if abs(freqs[q]) == 1:
                    w = -freqs[q]*nij                

                    cilm_phi[0,n+1,m] += phi[x,q,m,n].real*np.cos(w*t) + phi[x,q,m,n].imag*np.sin(w*t)
                    cilm_phi[1,n+1,m] += -phi[x,q,m,n].imag*np.cos(w*t) + phi[x,q,m,n].real*np.sin(w*t)

                    cilm_psi[0,n+1,m] += psi[x,q,m,n].real*np.cos(w*t) + psi[x,q,m,n].imag*np.sin(w*t) 
                    cilm_psi[1,n+1,m] += -psi[x,q,m,n].imag*np.cos(w*t) + psi[x,q,m,n].real*np.sin(w*t)

                    phi_eta = phi[x,q,m,n]*(n+1)*(n+2)*ho[x]/(radius**2.0 *w)
                    
                    cilm_eta[0,n+1,m] += -phi_eta.imag*np.cos(w*t) + phi_eta.real*np.sin(w*t) #+ psi[x,q,m,n].imag 
                    cilm_eta[1,n+1,m] += phi_eta.real*np.cos(w*t) - phi_eta.imag*np.sin(w*t) #- psi[x,q,m,n].real
                   
    PHI = MakeGridDH(cilm_phi, norm=3, sampling=2, lmax=60)
    PSI = MakeGridDH(cilm_psi, norm=3, sampling=2, lmax=60)
    ETA = MakeGridDH(cilm_eta, norm=3, sampling=2, lmax=60)

    lons = np.arange(0, 360, 360/np.shape(PHI)[1])
    colats = np.arange(0, 180, 180/np.shape(PHI)[0])
    colats[0] = 0.001

    XX, YY = np.meshgrid(np.radians(lons), np.radians(colats))

    gPHI = np.gradient(PHI, np.radians(colats), np.radians(lons),  edge_order=1)
    gPSI = np.gradient(PSI, np.radians(colats), np.radians(lons),  edge_order=1)

    dPHIdlon = gPHI[1]
    dPHIdlat = gPHI[0]

    dPSIdlon = gPSI[1]
    dPSIdlat = gPSI[0]

    U = ((dPHIdlon/np.sin(np.radians(colats[:,None])) - dPSIdlat)/radius).real
    V = ((dPHIdlat + dPSIdlon/np.sin(np.radians(colats[:,None])))/radius).real
    V[-1,:] *= 0.0                 
    U[-1,:] *= 0.0

    # MAG = np.sqrt(U**2 + V**2)

    # gU = np.gradient(U, np.radians(colats), np.radians(lons), edge_order=1)
    # gV = np.gradient(V, np.radians(colats), np.radians(lons),  edge_order=1)

    # dudlon = gU[1]
    # dudlat = gU[0]

    # dvdlon = gV[1]
    # dvdlat = gV[0]

    # adv_u = U / (np.sin(YY)*radius) * dudlon + V / radius * dudlat + U*V / (np.tan(YY) * radius)
    # adv_v = U / (np.sin(YY)*radius) * dvdlon + V / radius * dvdlat - U**2.0 / (np.tan(YY) * radius)
    # adv_u[-1,:] *= 0.0
    # adv_v[-1,:] *= 0.0

    # adv_u[0,:] *= 0.0
    # adv_v[0,:] *= 0.0

    # adv_u *= -1
    # adv_v *= -1


    # MAG_ADV = np.sqrt(adv_u**2 + adv_v**2)
    # # print(np.shape(adv))



    # u_norm = U/MAG
    # v_norm = V/MAG

    # v_adv_norm = adv_v/MAG_ADV
    # u_adv_norm = adv_u/MAG_ADV

    # # u_adv_total += u_adv_norm
    # # v_adv_total += v_adv_norm

    fig, axes = plt.subplots(nrows=3, figsize=(8,4*3.2))

    ax1, ax2, ax3 = axes

    skip = 6

    p1 = ax1.contourf(lons, 90-colats, U, 11)
    # ax1.quiver(lons[::skip], 90-colats[::skip], U[::skip,::skip], V[::skip,::skip])

    # print(np.shape(lons[::skip]))

    p2 = ax2.contourf(lons, 90-colats, V, 11)
    # ax2.quiver(lons[::skip], 90-colats[::skip], adv_u[::skip,::skip], adv_v[::skip,::skip])

    # p2 = ax2.contourf(lons, 90-colats, MAG_ADV)
    p3 = ax3.contourf(lons, 90-colats, ETA, 11)
    # ax3.contour(lons, 90-colats, (adv_u*u_norm + adv_v*v_norm)/MAG_ADV, levels=[0])

    cb = plt.colorbar(p1, ax=ax1, label='Flow speed $|\\vec{u}|$ [\\si{\\metre\\per\\second}]')
    cb = plt.colorbar(p2, ax=ax2, label='Advection magnitude $|-(\\vec{u}\cdot\\nabla)\\vec{u}|$')
    cb = plt.colorbar(p3, ax=ax3, label='$\\frac{[-(\\vec{u}\cdot\\nabla)\\vec{u}] \cdot \\hat{u}}{|(\\vec{u}\cdot\\nabla)\\vec{u}|}$')

    for ax in axes:
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

    fig.savefig("/home/hamish/Dropbox/Tests/res_flow_{:03d}.pdf".format(count), dpi=100, bbox_inches='tight')
    count += 1

# fig, ax3 = plt.subplots()
# p3 = ax3.contourf(lons, 90-colats, MAG, cmap=plt.cm.bwr)

# fig.savefig("/home/hamish/Dropbox/Tests/neG_T_AVG.PNG", dpi=200)
# ax3.contour(lons, 90-colats, MAG, levels=[0])

# fig, ax = plt.subplots()

# ax.semilogy(ho, diss)
# ax.scatter(ho[x], diss[x])

# plt.show()


# print(u_total/count)
# print(v_total/count)
# print(u_adv_total/count)
# print(v_adv_total/count)

# plt.quiver(lons, 90-colats, u_adv_total/count, v_adv_total/count)
# plt.show()
