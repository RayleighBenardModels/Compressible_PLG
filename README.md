# Playground for compressible Rayleigh-Bénard convection

This repository provides a collection of python scripts to model
compressible convection using [dedalus](http://dedalus-project.org).
The equation of state used in these scripts relates density to
specific entropy. Four levels of approximation are used:

* RB_FC.py: full compressible. This is the most complete set of
equations, with no approximation beyond the continuum physics one.
* RB_AA.py: Anelastic approximation.
* RB_ALA.py: Anelastic liquid approximation.
* RB_SCA.py: Simple compressible approximation.

Details of the theory and all the equations for these models can be
found in the paper

Thierry Alboussière, Jezabel Curbelo, Fabien Dubuffet, Stéphane
Labrosse and Yanick Ricard,
A playground for compressible natural convection with a nearly uniform
density,
*J. Fluid Mech.*, submitted.
  
