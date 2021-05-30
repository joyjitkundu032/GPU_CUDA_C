# GPU_CUDA_C

This prog simulates the system of hard rods on a 3D cubic lattice in grand canonical ensemble to study phase transitions. 

The code requires two input files consisting of the list of probabilities for open (accept*) and periodic (periodic*) boundary conditions.

The code has been parallelized using CUDA C to run on GPU. 

Here, the chemical potential $\mu$ is the control parameter that governs the packing fraction. The phase change occurs as one varies the packing fraction. 
