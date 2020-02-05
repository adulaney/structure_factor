# structure_factor
A module for computing static and dynamic structure factor from a GSD simulation script.

# Structure Factor Calculation

This module is used to compute dynamic and static structure factors from GSD simulation files.

## Requirements
-CMake >= 3.0
-C++11 capable compiler (tested with gcc 7.4.0)
-CUDA >= 8.0 (optional)

## Compiling
In the structure factor folder run

```
mkdir build
cd build
cmake ../
make -j4
```

This module can be compiled with CUDA support by running cmake with
```
cmake ../ -DENABLE_CUDA=ON
```

The module can also be built with verbose CUDA instructions using the ```-DCUDA_VERBOSE_BUILD``` flag, which is set to False by default.

## Examples
```
import structure_factor
import gsd.hoomd

s = gsd.hoomd.open('path_to_trajectory.gsd', mode='rb')

SSF = structure_factor.structure_class.ssf(s, compute_mode='gpu')
SSF.compute(max_kint=50, frame=0, single_vec=False, gpu_id=1)
```

## Static Structure Factor class
structure_class.ssf(traj,compute_mode='gpu')
- traj: GSD simulation trajectory
- compute_mode: (str) Determines whether calculation is handled on "cpu" or "gpu"
### Functions
ssf.compute(frame=0,max_kint=15, single_vec=True, gpu_id=0)
- frame: (int) Frame of trajectory to perform the calculation on
- max_kint: (int) Number of wavevectors to compute the structure factor over. Sampled using the Nyquist frequency.
-single_vec: (bool) Whether we wish to consider wavevectors in more than one dimension (i.e. k_x, k_y, k_z integer values)
-gpu_id: (int) The id of the gpu to use for computation if compute_mode="gpu"



# Acknowledgements
Many of the backend datastructures used come from the [HOOMD](https://hoomd-blue.readthedocs.io/en/stable/index.html) source code and as such I thank their developers greatly.






