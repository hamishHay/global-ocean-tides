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

plt.set_cmap('viridis')

alpha = 1e-11

den = 1e3
moon = 'europa'
h = 68103.27196342
# h = 10e3

den = 3e3
moon = 'io'
h = 46489.34492713

data = planetPy.database.Database()
moons = data.jupiter.moons
radius = moons[moon.capitalize()].radius



fig, ax = plt.subplots(figsize=(8,4))



# infile = h5py.File("LTE_solutions_alpha_" + str(alpha)+ ".h5", 'r')
# infile = h5py.File("europa_v1e19_a1e-11.h5", 'r')
# infile = h5py.File("europa_res_68km.h5", 'r')
# infile = h5py.File("res_io_64km.h5", 'r')
infile = h5py.File("io_res_46km.h5", 'r')

# infile = h5py.File("positive_test.h5", 'r')



freqs = list(infile[moon]['frequency'][:])
omega = list(infile[moon]['frequency qw'][:])

ho = infile[ moon ]['ocean thickness'][:]

phi = infile[moon]['velocity potential'][:]
psi = infile[moon]['stream function'][:]
# print(phi[0, 55])
diss = infile[moon]['dissipated power: ocean'][:]# + infile[moon]['dissipated power: shell'][:]


diss = np.sum(diss, axis=0)

infile.close()

n_max = len(phi[0, 0, 0, :])
n = np.arange(1, n_max+1, 1)

Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
Nnm2[0] = 1.0
Nnm1 = n*(n+1)/(2*n+1)*factorial(n+1)/factorial(n-1)

dt = 0.01
N = 10



idx = (np.abs(ho - h)).argmin()
x = idx

nij = 0.5* 2.05e-5
P = abs(2*np.pi/(4*nij))
times = np.arange(0, 1.0, dt)

MAX_VEL = 0
MAX_DISP = 0

FLUX = 0
avg_vel = np.zeros(len(freqs))
for time in times:
    t=(time+0.0)*P 
    
    FLUX_DT = 0

    cilm_phi = np.zeros((2, n_max+1, n_max+1))
    cilm_psi = np.zeros((2, n_max+1, n_max+1))
    cilm_eta = np.zeros((2, n_max+1, n_max+1))
    
    ETA = None
    for m in [0, 2]:
        for n in range(8):  
            for q in range(len(freqs)):
                if abs(freqs[q]):
                    w = omega[q]#-freqs[q]*nij                

                    cilm_phi[0,n+1,m] += phi[x,q,m,n].real*np.cos(w*t) + phi[x,q,m,n].imag*np.sin(w*t)
                    cilm_phi[1,n+1,m] += -phi[x,q,m,n].imag*np.cos(w*t) + phi[x,q,m,n].real*np.sin(w*t)

                    cilm_psi[0,n+1,m] += psi[x,q,m,n].real*np.cos(w*t) + psi[x,q,m,n].imag*np.sin(w*t) 
                    cilm_psi[1,n+1,m] += -psi[x,q,m,n].imag*np.cos(w*t) + psi[x,q,m,n].real*np.sin(w*t)

                    phi_eta = phi[x,q,m,n]*(n+1)*(n+2)*ho[x]/(radius**2.0 *w)
                    
                    cilm_eta[0,n+1,m] += -phi_eta.imag*np.cos(w*t) + phi_eta.real*np.sin(w*t) #+ psi[x,q,m,n].imag 
                    cilm_eta[1,n+1,m] += phi_eta.real*np.cos(w*t) - phi_eta.imag*np.sin(w*t) #- psi[x,q,m,n].real
                   
    PHI = MakeGridDH(cilm_phi, norm=3, sampling=2, lmax=20)
    PSI = MakeGridDH(cilm_psi, norm=3, sampling=2, lmax=20)
    ETA = MakeGridDH(cilm_eta, norm=3, sampling=2, lmax=20)

    xx = np.arange(0, 360, 360/np.shape(PHI)[1])
    yy = np.arange(0, 180, 180/np.shape(PHI)[0])
    yy[0] = 0.001

    gPHI = np.gradient(PHI, np.radians(yy), np.radians(xx))
    gPSI = np.gradient(PSI, np.radians(yy), np.radians(xx))

    dPHIdlon = gPHI[1]
    dPHIdlat = gPHI[0]

    dPSIdlon = gPSI[1]
    dPSIdlat = gPSI[0]

    U = ((dPHIdlon/np.sin(np.radians(yy[:,None])) - dPSIdlat)/radius).real
    V = ((dPHIdlat + dPSIdlon/np.sin(np.radians(yy[:,None])))/radius).real
    V[-1,:] *= 0.0                 
    U[-1,:] *= 0.0

    FLUX_DT += U**2.0 + V**2.0

    FLUX += abs(FLUX_DT) * dt*P

    MAX_DISP = max(MAX_DISP, np.amax(ETA))
    MAX_VEL = max(MAX_VEL, np.amax(np.sqrt(U**2.0 + V**2.0)))

FLUX *= den*abs(ho[x])*alpha/P

np.savetxt(moon+"_res_46km.txt", FLUX.T)

areas = radius**2.0 * np.sin(np.radians(yy)) * np.radians(xx[1]-xx[0])**2.0

print(MAX_VEL, MAX_DISP)

# print(E*np.pi * 2 * alpha * den * ho[x]/1e9, np.sum(FLUX*areas[:, None])/1e9)

print(np.sum(FLUX*areas[:, None])/1e9, np.sum(diss[x])/1e9,  np.sum(diss[x])/(4*np.pi*radius**2.0))


p1 = ax.contourf(xx, 90-yy, FLUX, 11)
plt.colorbar(p1, ax=ax, label='Heat Flux')

# p1 = ax.quiver(lons, 90-colats, U.real, V.real)

fig.savefig("/home/hamish/Dropbox/Tests/res_flow.pdf", dpi=100, bbox_inches='tight')

# plt.quiver(lons, 90-colats, u_adv_total/count, v_adv_total/count)
# plt.show()

