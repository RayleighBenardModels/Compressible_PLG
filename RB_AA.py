"""

Anelastic Approximation AA

Simulation script for 2D Rayleigh-Benard equation, infinite Pr.

"""
# input parameters

Ra = rayleighnumber     # superadiabatic Rayleigh number
Di = dissipationnumber  # dissipation number
tps = tempsfinal        # substitute for the desired final time of the simulation
tstep = 0.001/Ra**(2/3)     # initial timestep
stepsimu = 0.02/Ra**(2/3)   # timestep for solution analysis

N = IntegN  # value of the parameter n of the Equation of state

# a few parameters (a to f) that will be used to compute the initial profiles
a = 0.0
b = 0.0
c = 0.0
d = 0.0
e = 0.0
f = 0.0

import os
import numpy as np
import time
import sys
import h5py
#import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools

import time

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

#Aspect ratio 
Lx, Lz = (5.65685425, 1.)   # 4 sqrt(2) = 5.65685425 corresponds to two periods of the eigenvector at threshold
nx, nz = (kx, kz)           # substitute kx and kz for the number of Fourier and Chebyshev modes

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z',nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# RB equations

problem = de.IVP(domain, variables=['P','u','w','uz','wz','T','Tz'])    # pressure, both components of the velocity vector, their z-derivative with respect to z, superadiabatic temperature and its derivative with respect to z are the primary variables.
problem.meta['P','u','w','uz','T']['z']['dirichlet'] = True

problem.parameters['R'] = Ra*(N+1)
problem.parameters['D'] = Di*(N+1)
problem.parameters['n'] = N
problem.parameters['Lxb'] = Lx
problem.parameters['Lzb'] = Lz

problem.substitutions['diss'] = "2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2"  # local viscous dissipation
problem.substitutions['mT'] = "integ(integ(T,'x'),'z')/(Lxb*Lzb)"       # mean superadiabatic temperature over the whole domain
problem.substitutions['Ta'] = "1.0-D/(n+1)*z"   # adiabatic temperature profile
problem.substitutions['meanrho'] = "integ(integ(-T/Ta+P/(n+1)/Ta,'x'),'z')/(Lxb*Lzb)"   # mean density departure from the adiabat
problem.substitutions['Tameanrho'] = "integ(integ(-T+P/(n+1),'x'),'z')/(Lxb*Lzb)"       # mean value of the product Ta by rho

problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) - 1/(n+1)*dt(P)  = - u*dx(T) - w*Tz + 1/(n+1)*u*dx(P) + 1/(n+1)*w*dz(P) - D/(n+1)*w*T/Ta + D/(n+1)**2*w*P/Ta + D/R*diss")    # thermal equation
problem.add_equation("dx(P) - D/R*(dx(dx(u)) + dz(uz)) = 0") # Stokes equation in the x direction
problem.add_equation("dz(P) - D/R*(dx(dx(w)) + dz(wz)) - D/(n+1)*T/Ta + D/(n+1)**2*P/Ta = 0")   # Stokes equation in the z direction
problem.add_equation("uz - dz(u) = 0")  # definition of z-derivative of u
problem.add_equation("wz - dz(w) = 0")  # definition of z-derivative of w
problem.add_equation("Tz - dz(T) = 0")  # definition of z-derivative of T
problem.add_equation("dx(u) + wz = 0")  # mass conservation

# boundary conditions


problem.add_bc("left(uz) = 0.")     # no-stress
problem.add_bc("right(uz) = 0.", condition="(nx != 0)") # no-stress
problem.add_bc("right(u) = 0.", condition="(nx == 0)")  # zero average
problem.add_bc("left(w) = 0.")                          # non-permeable
problem.add_bc("right(w) = 0.0", condition="(nx != 0)") # non-permeable
problem.add_bc("left(P)-right(P) = 0.0", condition="(nx == 0)") # ensures global mass conservation
problem.add_bc("left(T) = 1.0/2")   # hot temperature at the bottom
problem.add_bc("right(T) = -1.0/2") # cold temperature at the top

# time stepping

ts = de.timesteppers.RK443

# Build solver
solver = problem.build_solver(ts)

# initial conditions
x = domain.grid(0)
z = domain.grid(1)
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']
T = solver.state['T']
Tz = solver.state['Tz']
P = solver.state['P']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

zb, zt = z_basis.interval
pert =  1e-1 * noise * (zt - z) * (z - zb)

u['g'] = 0.0        # initial x-velocity
w['g'] = 0.0        # initial z-velocity
T['g'] = - z + pert # initial conduction profile + noise

a=(-Di/24.0*(2-1/(N+1))-Di**3/1920*(4-1/(N+1))*(3-1/(N+1))*(2-1/(N+1)))*(N+1)/(1+Di**2/24*(2-1/(N+1))*(1-1/(N+1))+Di**4/1920*(4-1/(N+1))*(3-1/(N+1))*(2-1/(N+1))*(1-1/(N+1)))
b=-Di*a/(N+1)
c=Di*b/2.0*(1-1/(N+1))-Di/2.0
d=Di*c/3.0*(2-1/(N+1))
e=Di*d/4.0*(3-1/(N+1))
f=Di*e/5.0*(4-1/(N+1))
g=Di*f/6.0*(5-1/(N+1))
h=Di*g/7.0*(6-1/(N+1))
P['g'] = a + b * z + c * z**2 + d * z**3 + e * z**4 + f * z**5 + g * z**6   # initial hydrostatic pressure (polynomial approximation at degree 6)
#
u.differentiate('z',out=uz)
w.differentiate('z',out=wz)
T.differentiate('z',out=Tz)

# integration parameters and CFL

solver.stop_sim_time = tps
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = tstep
cfl = flow_tools.CFL(solver,initial_dt,safety=0.9, max_change=1.05, min_change=0.95)
cfl.add_velocities(('u','w'))

# analysis

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=stepsimu, max_writes=1000)

# The quantities defined below (and in snapshots) are stored for post
# analysis in h5 files.
# They are listed for indication, others can be added, some can be removed,
# depending on the interest of each user.

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz
solver.evaluator.vars['Ra'] = Ra
solver.evaluator.vars['Di'] = Di
solver.evaluator.vars['N'] = N
solver.evaluator.vars['tps'] = tps
solver.evaluator.vars['tstep'] = tstep

# fields scalars
analysis.add_task("N", name='N')
analysis.add_task("Ra", name='Ra')
analysis.add_task("Di", name='Di')
analysis.add_task("tps", name='tps')
analysis.add_task("tstep", name='tstep')

# fields t
analysis.add_task("integ(integ(Di*(- w*T/Ta + 1/(n+1)*w*P/Ta)+Di/Ra*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2), 'x'), 'z')/(Lx*Lz)", name='integNet')
analysis.add_task("integ(integ(Di/Ra*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2), 'x'), 'z')/(Lx*Lz)", name='integDiss')
analysis.add_task("integ(integ(T,'x'),'z')/(Lx*Lz)", name='Tmean')
analysis.add_task("integ(integ(D/R*diss, 'x'), 'z') / (Lx * Lz)", name='Total_diss')
analysis.add_task("integ(integ(-T/Ta+P/(n+1)/Ta,'x'),'z')/(Lxb*Lzb)", name='meanrho')
analysis.add_task("integ(integ(T*w,'x'),'z')/(Lxb*Lzb)", name='integTw')
analysis.add_task("integ(integ(P*w,'x'),'z')/(Lxb*Lzb)", name='integPw')
analysis.add_task("integ(integ(-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x'),'z')/(Lxb*Lzb)", name='viscHeatFlux')
analysis.add_task("integ(abs(integ(diss/R,'x')/integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')-1/(1-D*z)),'z')/Lzb", name='abs_diss_dist')
analysis.add_task("integ(abs(integ(w*(T-1/(n+1)*P)-Tz,'x')/integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')-1),'z')/Lzb", name='abs_flux_dist')

# fields z, t
analysis.add_task("integ(T,'x')/Lx", name='T profile')
analysis.add_task("integ(u**2+w**2,'x')/Lx", name='KE profile')
analysis.add_task("integ(D/R*diss, 'x') / Lx ", name='Total_diss_profile')
analysis.add_task("integ(D/R*diss/Ta, 'x') / Lx ", name='visc_entropysource_profileTa')
analysis.add_task("integ(-Tz,'x')/Lx", name='heat_flux_cond')
analysis.add_task("integ(w*(T+n/(n+1)*P),'x')/Lx", name='heat_flux_enthalpyTa')
analysis.add_task("integ(w*(T-1/(n+1)*P),'x')/Lx", name='heat_flux_entropyTa')
analysis.add_task("integ(w*T,'x')/Lx", name='heat_flux_Ta_Tw')
analysis.add_task("integ(w*P,'x')/Lx", name='heat_flux_Ta_Pw')
analysis.add_task("integ(-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux_viscwork')
analysis.add_task("integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux')
analysis.add_task("integ(w*P-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='funF')
analysis.add_task("integ((T-1/(n+1)*P)*w/Ta-1/R*diss, 'x') / Lx ", name='dzfunF')
analysis.add_task("integ(w*(T-1/(n+1)*P),'x')/integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')", name='relative_entropy_flux')
analysis.add_task("integ(D/R*diss,'x')/integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')", name='relative_diss')

#snapshots

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10*stepsimu, max_writes=500)

snapshots.add_task('t')
snapshots.add_task('x')
snapshots.add_task('z')
snapshots.add_task('T')
snapshots.add_task('Tz')
snapshots.add_task('P')
snapshots.add_task('u')
snapshots.add_task('uz')
snapshots.add_task('w')
snapshots.add_task('wz')
snapshots.add_task("diss", name='Diss')

# main loop

start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)

end_time = time.time()
