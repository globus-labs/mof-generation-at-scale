# Configuration Files

Configuration files for running MOFA in different modes
on different computers.

Configuration requires a series of Python files:

1. An HPC config as a file which instantiates a subclass of `HPCConfig` and names the variable `hpc_config`.

MOFA executes then reads each of the named variables out of the configuration files.
