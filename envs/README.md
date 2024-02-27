Environment files for different resources

## Building LAMMPS on Polaris

Follow the instructions [from ALCF for building LAMMPS with Kokkos](https://github.com/argonne-lcf/GettingStarted/tree/master/Applications/Polaris/LAMMPS)

```bash
#! /bin/bash

# Make the build environment
module use /soft/modulefiles
module load cudatoolkit-standalone/11.8.0
module load kokkos
module list

# Build
cd src
cd MAKE/MACHINES/
wget -c https://github.com/argonne-lcf/GettingStarted/raw/master/Applications/Polaris/LAMMPS/Makefile.polaris_nvhpc_kokkos
cd ../..
pwd
#make yes-most
make polaris_nvhpc_kokkos -j 16
```
