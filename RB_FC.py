"""

This is a full-compressible version FC

Simulation script for 2D Rayleigh-Benard equation, infinite Pr.

"""
# input parameters

tps = tempsfinal    # substitute for the desired final time of the simulation
Ra = rayleighnumber     # superadiabatic Rayleigh number
Di = dissipationnumber  # dissipation number
eps = epsilon           # superadiabatic temperature difference
N = valueN              # value of the parameter n of the Equation of state
tstep = 0.5*eps*Di/Ra/(N+1)**2  # timestep limited by viscous relaxation timescale. The factor 0.5 is useful in the beginning of convection characterized by a peak in viscous dissipation. It can be removed at longer times.

# a few parameters (a to f) that will be used to compute the initial profiles
a = 1.0
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

from dedalus import public as de

import time

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)


#Aspect ratio 
Lx, Lz = (5.65685425, 1.)  # 4 sqrt(2) = 5.65685425 corresponds to two periods of the eigenvector at threshold
nx, nz = (kx, kz)   # substitute kx and kz for the number of Fourier and Chebyshev modes 

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z',nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# Rayleigh-Benard

problem = de.IVP(domain, variables=['rho','u','w','uz','wz','T','Tz'])  # density, both components of the velocity vector, their z-derivative with respect to z, temperature and its derivative with respect to z are the primary variables. 
problem.meta['u','w','uz','wz','T','Tz']['z']['dirichlet'] = True

problem.parameters['R'] = Ra
problem.parameters['D'] = Di
problem.parameters['ep'] = eps
problem.parameters['n'] = N
problem.parameters['Lxb'] = Lx
problem.parameters['Lzb'] = Lz

problem.substitutions['diss'] = "2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2-2/3*(dx(u)+wz)**2"    # local viscous dissipation
problem.substitutions['divu'] = "dx(u)+wz"  # divergence of the velocity field

# governing equations

problem.add_equation("dt(rho) = -u*dx(rho)-w*dz(rho)-rho*divu")     # mass conservation
problem.add_equation("dx(dx(T)) + dz(Tz) - (n+1)*divu = - (n+1)*divu + (n+1)*rho**(n+1)*T*divu - ep*D/R*diss")  # thermal equation
problem.add_equation(" dx(dx(u)) + dz(uz) + (dx(dx(u))+dx(wz))/3 - R/ep/D*((n+1)**2*dx(rho)+(n+1)*dx(T)) = - R/ep/D*((n+1)**2*dx(rho)+(n+1)*dx(T)) + R/ep/D*rho**n*((n+1)**2*T*dx(rho)+(n+1)*rho*dx(T))") # Stokes equation in the x direction
problem.add_equation(" dx(dx(w)) + dz(wz) + (dx(uz)+dz(wz))/3 - R*(n+1)/ep*rho - R/ep/D*((n+1)**2*dz(rho)+(n+1)*Tz) = - R/ep/D*((n+1)**2*dz(rho)+(n+1)*Tz) + R/ep/D*rho**n*((n+1)**2*T*dz(rho)+(n+1)*rho*Tz)") # Stokes equation in the z direction
problem.add_equation("uz - dz(u) = 0")  # definition of z-derivative of u
problem.add_equation("wz - dz(w) = 0")  # definition of z-derivative of w
problem.add_equation("Tz - dz(T) = 0")  # definition of z-derivative of T

# boundary conditions

problem.add_bc("left(uz) = 0.")     # no-stress
problem.add_bc("right(uz) = 0.", condition="(nx != 0)")     # no-stress 
problem.add_bc("right(u) = 0.", condition="(nx == 0)")      # zero average 
problem.add_bc("left(w) = 0.")                              # non-permeable
problem.add_bc("right(w) = 0.")                             # non-permeable
problem.add_bc("left(T) = 1+D/2+ep/2")      # hot temperature at the bottom
problem.add_bc("right(T) = 1-D/2-ep/2")     # cold temperature at the top

# time stepping

ts = de.timesteppers.RK111

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
rho = solver.state['rho']


# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

zb, zt = z_basis.interval
pert =  1e-6 * noise * (zt - z) * (z - zb) 

# loop to reach an approximate initial hydrostatic conduction profile
for i in range(0,20):
    b=a/(N+1)*(Di+eps-Di*a**(-N))
    c=b/2.0/(N+1)*((N+2)*(Di+eps)-(1-N)*Di*a**(-N))
    d=c/3.0/(N+1)*((2*N+3)*(Di+eps)-(1-N)*Di*a**(-N))-Di/6.0/(N+1)*(1-N)*(-N)*b**2*a**(-1-N)
    e=d/4.0/(N+1)*((3*N+4)*(Di+eps)-(1-N)*Di*a**(-N))-Di/4.0/(N+1)*((1-N)*(-N)*b*c*a**(-1-N)+(1-N)*(-N)*(-1-N)/6.0*b**3*a**(-2-N))
    f=e/5.0/(N+1)*((4*N+5)*(Di+eps)-(1-N)*Di*a**(-N))-Di/5.0/(N+1)*((1-N)*(-N)*(c**2*a**(-1-N)+2*b*d*a**(-1-N))+(1-N)*(-N)*(-1-N)/2.0*b**2*c*a**(-2-N)+(1-N)*(-N)*(-1-N)*(-2-N)/24.0*b**4*a**(-3-N))
    a=1-c/12.0-e/80.0

rho['g'] = a + b * z + c * z**2 + d * z**3 + e * z**4 + f * z**5    # initial density profile
u['g'] = 0.0    # initial x-velocity
w['g'] = 0.0    # initial z-velocity
T['g'] = 1.0 - Di*z - eps*z + pert # initial conduction profile + noise
u.differentiate('z',out=uz)
w.differentiate('z',out=wz)
T.differentiate('z',out=Tz)

# integration parameters and CFL

solver.stop_sim_time = tps
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = tstep

# analysis

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.002, max_writes=round(500*tps))

# The quantities defined below (and in snapshots) are stored for post 
# analysis in h5 files. 
# They are listed for indication, others can be added, some can be removed,
# depending on the interest of each user. 

analysis.add_task('T')
analysis.add_task('u')
analysis.add_task('w')

solver.evaluator.vars['N'] = N
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz
solver.evaluator.vars['Ra'] = Ra
solver.evaluator.vars['Di'] = Di 
solver.evaluator.vars['eps'] = eps
solver.evaluator.vars['tps'] = tps
solver.evaluator.vars['tstep'] = tstep

# fields scalars
analysis.add_task("N", name='N')
analysis.add_task("Ra", name='Ra')
analysis.add_task("Di", name='Di')
analysis.add_task("eps", name='eps')
analysis.add_task("tps", name='tps')
analysis.add_task("tstep", name='tstep')

# fields x, z, t
analysis.add_task("eps*D/R*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2-2/3*(dx(u)+wz)**2)", name='Diss')
analysis.add_task("-(dx(u)+wz)*(n+1)*rho**(n+1)*T", name='pow')
analysis.add_task("T-1.0+D/(n+1)*z", name='Tsa')

# fields t
analysis.add_task("integ(integ(eps*D/R*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2-2/3*(dx(u)+wz)**2)-(dx(u)+wz)*(n+1)*rho**(n+1)*T, 'x'), 'z')/(Lx*Lz)", name='IntegNet')
analysis.add_task("integ(integ(eps*D/R*(2*(dx(u))**2+(dx(w)+uz)**2+2*(wz)**2-2/3*(dx(u)+wz)**2), 'x'), 'z')/(Lx*Lz)", name='IntegDiss')
analysis.add_task("integ(integ((n+1)*T*rho**(n+1)*divu,'x'),'z')/(Lx*Lz)", name='Pdivu')
analysis.add_task("integ(integ(T,'x'),'z')/(Lx*Lz)", name='Tmean')
analysis.add_task("integ(integ(T-1.0+D/(n+1)*z,'x'),'z')/(eps*Lx*Lz)", name='Tsamean')
analysis.add_task("integ(integ(D/R*diss, 'x'), 'z') / (Lx * Lz)", name='Total_diss')


# fields z, t
analysis.add_task("integ(T,'x')/Lx", name='T profile')
analysis.add_task("integ(T-1.0+D/(n+1)*z,'x')/Lx", name='superadiabTprofile')
analysis.add_task("integ(u**2+w**2,'x')/Lx", name='KE profile')
analysis.add_task("integ(D/R*diss, 'x') / Lx ", name='Total_diss_profile')
analysis.add_task("integ(-Tz,'x')/Lx", name='heat_flux_cond')
analysis.add_task("integ(rho*w*(n+1)*T*rho**n,'x')/Lx", name='heat_flux_enthalpy')
analysis.add_task("integ(-eps*D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux_viscwork')
analysis.add_task("integ(-Tz+rho*w*(n+1)*T*rho**n-eps*D/R*(u*(uz+dx(w))+w*(4/3*wz-2/3*dx(u))),'x')/Lx", name='heat_flux')


snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.005, max_writes=round(500*tps))

snapshots.add_task("T-1.0+D/(n+1)*z", name='Tsa')
snapshots.add_task('t')
snapshots.add_task('x')
snapshots.add_task('z')
snapshots.add_task('rho')
snapshots.add_task('T')
snapshots.add_task('Tz')
snapshots.add_task('u')
snapshots.add_task('uz')
snapshots.add_task('w')
snapshots.add_task('wz')
snapshots.add_task("diss", name='Diss')

# main loop

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = tstep
    solver.step(dt)

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

