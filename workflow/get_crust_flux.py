import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.misc import factorial
from scipy.interpolate import interp1d
from pyshtools.legendre import PLegendreA_d1
from pyshtools.expand import SHExpandDH
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
h = 15e3
# h = 68103.27196342

# den = 3e3
# moon = 'io'
# h = 80244.83145156

data = planetPy.database.Database()
moons = data.jupiter.moons
radius = moons[moon.capitalize()].radius

fig, ax = plt.subplots(figsize=(8,4))



# infile = h5py.File("LTE_solutions_alpha_" + str(alpha)+ ".h5", 'r')
infile = h5py.File("spring_test.h5", 'r')
# infile = h5py.File("eur_res_h68.1km_v1e18_a11.h5", 'r')
# infile = h5py.File("io_heat_80.h5", 'r')

# infile = h5py.File("m2_test.h5", 'r')



freqs = list(infile[moon]['frequency'][:])
# omegas = list(infile[moon]['frequency q'][:])

# print(freqs)

# print(omegas)

ho = infile[ moon ]['ocean thickness'][:]


phi = infile[moon]['velocity potential'][:]
psi = infile[moon]['stream function'][:]
springs = infile[moon]['spring constant'][:]

diss = infile[moon]['dissipated power: ocean'][:] + infile[moon]['dissipated power: shell'][:]

diss = np.sum(diss, axis=0)
infile.close()

n_max = len(phi[0, 0, 0, :])
n = np.arange(1, n_max+1, 1)

Nnm0 = n*(n+1)/(2*n+1)*factorial(n)/factorial(n)
Nnm2 = n*(n+1)/(2*n+1)*factorial(n+2)/factorial(n-2)
Nnm2[0] = 1.0

dt = 0.01
N = 5

lons = np.arange(0, 360., N, dtype=float)
colats = np.arange(0.0+N, 180., N, dtype=float)
# colats[0] = 0.001
# colats[-1] -= 0.001
areas = radius**2.0 * np.sin(np.radians(colats)) * np.radians(N)**2.0

idx = (np.abs(ho - h)).argmin()

x = idx+1
# x = 0
# print(ho[x])

dPm0 = np.zeros((len(colats), n_max))
dPm1 = np.zeros((len(colats), n_max))
dPm2 = np.zeros((len(colats), n_max))
Pm0 = np.zeros((len(colats), n_max))
Pm1 = np.zeros((len(colats), n_max))
Pm2 = np.zeros((len(colats), n_max))
for i in range(len(colats)):
    P, dPdTheta = PLegendreA_d1(n_max, np.cos( np.radians(colats[i]) ) )

    dPdTheta *= -np.sin( np.radians(colats[i]) )
    for j in range(1, n_max):
        l = j

        m=0
        m0_indx = int(l*(l+1)/2+m)
        dPm0[i, j-1] = dPdTheta[ m0_indx ]
        Pm0[i, j-1] = P[ m0_indx ]

        m=1
        m0_indx = int(l*(l+1)/2+m)
        dPm1[i, j-1] = dPdTheta[ m0_indx ]
        Pm1[i, j-1] = P[ m0_indx ]

        m=2
        m2_indx = int(l*(l+1)/2+m)
        dPm2[i, j-1] = dPdTheta[ m2_indx ]
        Pm2[i, j-1] = P[ m2_indx ]

XX, YY = np.meshgrid(np.radians(lons), np.radians(colats))

MAG = np.zeros(np.shape(XX), dtype=np.float)

F_crust = np.zeros(np.shape(XX), dtype=np.float)

nij = 0.5*2.05e-5#5.892342791791881e-06#2.05e-5*0.5
P = 2*np.pi/nij
dt = 0.05
times = np.arange(0, 2.0, dt)
# times = [0]
count = 0
# spring = 2.0e-6 + 1e-10j
for time in times:
    q_load = np.zeros(np.shape(XX), dtype=np.float)
    eta_dot = np.zeros(np.shape(XX), dtype=np.float)
    eta = np.zeros(np.shape(XX), dtype=np.float)
    q_eta_dot = np.zeros(np.shape(XX), dtype=np.float)

    t = time*P

    # mm = 1
    for n in range(4):
        for m in [0, 2]:
            for q in range(len(freqs)):
                if abs(freqs[q])==2:
                    qw = freqs[int(q)]*nij
         
                    nn = n+1


                    
                    eta_nm = -nn*(nn+1)/radius**2 * ho[x]/qw * (phi[x,q,m,n].imag - 1j*phi[x,q,m,n].real)
                    spring = springs[q, n]

                    if m == 0:
                        Y_nm = Pm0[:,n][:, np.newaxis]*np.exp(1j * m * XX) 
                        # spring = springs[q, n]/
                    elif m==1:
                        Y_nm = Pm1[:,n][:, np.newaxis]*np.exp(1j * m * XX)
                    else:
                        Y_nm = Pm2[:,n][:, np.newaxis]*np.exp(1j * m * XX)
                        # spring = springs[q, sn]/np.sqrt(4*np.pi * Nnm2[n] / (nn*(nn+1)) )
                    


    # q_load *= 1e3*1.3

    # F_crust += abs(q_load * eta_dot)*dt*P
    # F_crust += abs(eta_dot*q_load)*dt
    F_crust += abs(eta)

# F_crust /= len(times)
# F_crust *= 1./areas[:, None]
    
p1 = ax.contourf(lons, 90-colats, F_crust)
# ax3.contour(lons, 90-colats, (adv_u*u_norm + adv_v*v_norm)/MAG_ADV, levels=[0])
E_crust = np.sum(F_crust * areas[:, None])

print(np.log10(E_crust), np.mean(np.mean(F_crust)))
cb = plt.colorbar(p1, ax=ax)
# cb = plt.colorbar(p2, ax=ax2, label='Advection magnitude $|(\\vec{u}\cdot\\nabla)\\vec{u}|$')
# cb = plt.colorbar(p3, ax=ax3, label='$\\frac{[(\\vec{u}\cdot\\nabla)\\vec{u}] \cdot \\hat{u}}{|(\\vec{u}\cdot\\nabla)\\vec{u}|}$')

# # # cb.ax.get_yticks()
# # # print([item.get_text() for item in cb.ax.get_yticklabels()])
# # # labels = ["{:.2f}".format(item) for item in cb.ax.get_yticks().tolist()]
# # # cb.ax.set_yticklabels(labels)
# # #
fig.savefig("/home/hamish/Dropbox/Tests/disp.pdf", dpi=100, bbox_inches='tight')
count += 1

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

