# CUDA collo


This is the GPU parallel version of the collocation integrator for ODEs (ordinary differential equations).
Its primary design feature is to allow solving ODEs with different initial conditions or parameters simultaneously.
The parallelism level of the code is 1 integrator instance per thread.

For example, this can be used to get thousands of possible solutions to create a statistic via the Monte Carlo method.


Non-parallel version can be found here: https://github.com/shvak/collo

## How to use it

Define constants `system_order` and `method_stage` in main.cpp. System order is the number of equations of your ODE system.
and the length of the input vector. `method_stage` sets integration coefficients depending on your desired order of error.



You also need to set a right-hand side of your differential equation  *y' = f(t, y)*. You can do this by importing a .cuh header.
containing a function named `rhs`. This function should be a CUDA device function with these input parameters:
`__device__ void rhs(double *y, double *yd, double t, double *par, size_t step, size_t iternum)`

* `*y` -- input vector with length = `system_order`
* `*yd` -- output derivative vector with length = `system_order`
* `t` -- time of integration
* `par` -- vector of additional parameters
* `step` -- current step number
* `iternum` -- current iteration number

After you define your right-hand side, you can set your initial data. 
In `main.cpp` set the initial vector y that has a length of `system_order*blocks*threads`. Here, every `system_order` values would be allocated to an individual thread.
Then you can launch the program.  All additional parameters are set in `parameters.json` and do not require you to recompile code after you change them.

parameters.json values:
* `maxstep` -- total number of steps
* `dt` -- timestep value
* `skipstep` -- amount of steps to skip while writing results. (For example, if `skipstep`=10 then the output would be written every 10 steps.)
* `blocks` and `threads` -- CUDA numbers of blocks and threads to use. The total number of integrator instances would be equal to `blocks*threads`.
For performance improvement, try to keep the number of threads to be a multiple of 32. Maximum number depends on your GPU. If your ODE is computationally difficult, try lowering the amount of threads.
* `par_length` -- length of additional parameter vector (should always be 1 or greater).

### Usage

While using it, be careful with the size of `maxstep`. Integration results are stored in the GPU memory and are copied only after the integration ends.
Total size of allocated result array would be `sizeof(numeric_type)*maxstep/skipstep * system_order * blocks * threads`. If maxstep/skipstep is set too high, you may run out of memory and encounter a runtime error or segmentation fault.









  
 




