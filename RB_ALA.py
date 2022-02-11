"""

Anelastic Liquid Approximation ALA

Simulation script for 2D Rayleigh-Benard equation, inf Pr.

"""
# input parameters

Ra = rayleighnumber     # superadiabatic Rayleigh number
Di = dissipationnumber  # dissipation number
N = IntegN              # value of the parameter n of the Equation of state

tps = tempsfinal        # substitute for the desired final time of the simulation
tstep = 0.001/Ra**(2/3) # initial timestep

import os
import numpy as np
import time
import sys
import h5py

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
problem.substitutions['Ta'] = "1.0-D/(n+1)*z"                           # adiabatic temperature profile
problem.substitutions['meanrho'] = "integ(integ(-T/Ta+P/(n+1)/Ta,'x'),'z')/(Lxb*Lzb)"   # mean density departure from the adiabat
problem.substitutions['Tameanrho'] = "integ(integ(-T+P/(n+1),'x'),'z')/(Lxb*Lzb)"       # mean value of the product Ta by rho
problem.substitutions['unsTa'] = "integ(integ(1/Ta,'x'),'z')/(Lxb*Lzb)" # mean value of 1/Ta


problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) = - u*dx(T) - w*Tz - D/(n+1)*w*T/Ta + D/R*diss")     # thermal equation
problem.add_equation("dx(P) - D/R*(dx(dx(u)) + dz(uz)) = 0")    # Stokes equation in the x direction
problem.add_equation("dz(P) - D/R*(dx(dx(w)) + dz(wz)) - D/(n+1)*T/Ta = 0") # Stokes equation in the z direction
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
problem.add_bc("left(P) = 0.0", condition="(nx == 0)")  # zero average pressure
problem.add_bc("right(P) = 0.0", condition="(nx == 0)") # zero average pressure
problem.add_bc("right(T) = 0.0", condition="(nx != 0)") # isothermal boundary
problem.add_bc("left(T) = 0.0", condition="(nx != 0)")  # isothermal boundary
problem.add_bc("left(T)-right(T) = 1.0", condition="(nx == 0)") # temperature difference imposed between bottom and top boundaries

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
pert =  1e-6 * noise * (zt - z) * (z - zb)

u['g'] = 0.0    # initial x-velocity
w['g'] = 0.0    # initial z-velocity
K=1/Di+1/np.log((1-Di/2)/(1+Di/2));
T['g'] = K - z + pert   # initial conduction profile + noise


a=0.0;
b=Di*K;
c=Di/2*(Di*K-1);
d=Di**2/3*(Di*K-1);
e=Di**3/4*(Di*K-1);
f=Di**4/5*(Di*K-1);
g=Di**5/6*(Di*K-1);
h=Di**6/7*(Di*K-1);
P['g'] = a + b * z + c * z**2 + d * z**3 + e * z**4 + f * z**5 + g * z**6 + h * z**7 # initial hydrostatic pressure (polynomial approximation at degree 7)

u.differentiate('z',out=uz)
w.differentiate('z',out=wz)
T.differentiate('z',out=Tz)



# integration parameters and CFL

solver.stop_sim_time = tps
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = tstep
tstep = 0.001/Ra**(2/3)
stepsimu = 0.02/Ra**(2/3)
cfl = flow_tools.CFL(solver,initial_dt,safety=0.90, max_change=1.001, min_change=0.95)
cfl.add_velocities(('u','w'))

# analysis

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=stepsimu, max_writes=round(1000*tps))

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

# fields x, z, t
analysis.add_task('Di/Ra*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2)', name='Diss')
analysis.add_task('Di*(- w*T/Ta + 1/(n+1)*w*P/Ta)', name='pow')

# fields t
analysis.add_task("integ(integ(Di*(- w*T/Ta + 1/(n+1)*w*P/Ta)+Di/Ra*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2), 'x'), 'z')/(Lx*Lz)", name='integNet')
analysis.add_task("integ(integ(Di/Ra*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2), 'x'), 'z')/(Lx*Lz)", name='integDiss')
analysis.add_task("integ(integ(T,'x'),'z')/(Lx*Lz)", name='Tmean')
analysis.add_task("integ(integ(D/R*diss, 'x'), 'z') / (Lx * Lz)", name='Total_diss')
analysis.add_task("integ(integ(-T/Ta+P/(n+1)/Ta,'x'),'z')/(Lxb*Lzb)", name='meanrho')
analysis.add_task("integ(integ(T*w,'x'),'z')/(Lxb*Lzb)", name='integTw')
analysis.add_task("integ(integ(P*w,'x'),'z')/(Lxb*Lzb)", name='integPw')

# fields z, t
analysis.add_task("integ(T,'x')/Lx", name='T profile')
analysis.add_task("integ(u**2+w**2,'x')/Lx", name='KE profile')
analysis.add_task("integ(D/R*diss, 'x') / Lx ", name='Total_diss_profile')
analysis.add_task("integ(-Tz,'x')/Lx", name='heat_flux_cond')
analysis.add_task("integ(w*(T+n/(n+1)*P),'x')/Lx", name='heat_flux_enthalpyTa')
analysis.add_task("integ(w*T,'x')/Lx", name='heat_flux_Ta_Tw')
analysis.add_task("integ(w*P,'x')/Lx", name='heat_flux_Ta_Pw')
analysis.add_task("integ(-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux_viscwork')
analysis.add_task("integ(-Tz+w*(T+n/(n+1)*P)-D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux')

#snapshots

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10*stepsimu, max_writes=round(500*tps))

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



