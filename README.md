# MUQ Example #

Just a small example, playing around with MUQ. We try to find some parameters of an ODE model.

## ODE Model ##
We solve the ODE problem $u^{\prime\prime} = - \omega^2 u$ with initial conditions $u(0) = \alpha$, $u^\prime(0) = 0$
The true solution is $u(t) = \alpha \cos(\omega t)$.

The goal is to invert for parameters $\alpha, \omega$.

## Structure of the repository ##
* `analysis`: plot script for the 2d density
* `doctest`: external include for doctest
* `src/MPI`: Some wrappers for MPI functionality
* `src/ODEModel`: Simple implicit Euler solver for the ODE problem above
* `src/UQ`: Everyting for uncertainty quantification
* `src/generate_true_solution.py`: generates the function $u$ with some fixed parameters $\alpha, \omega$
* `src/main.cpp`: main script, which starts the UQ runner.
* `tests`: Contains unit tests, probably not working

## How to run ##
* `mkdir build && cd build`
Note: I use muq2 with commit `7fafda2`.
* `cmake ..` 
* `make -j 8`
* `python ../src/generate_true_solution.py`
* `mpirun ./main`
* `python ../analyis/plot_density.py samples.h5`

