Running MOFA
============

MOFA is designed to, eventually, deploy multiple workflow styles but only supports one now:
running all components in a single batch job on a supercomputer.

The Run Script
--------------

.. note::

    We are working to break major blocks of configuration into configuration files
    rather than a large number of CLI options.

The ``run_parallel_workflow.py`` script deploys MOFA onto an HPC system.

**TBD**

Configuration
-------------

Configuration is broken into a few files

GenAI Components
++++++++++++++++

**TBD**

HPC Layout
++++++++++

The :class:`~mofa.hpc.config.HPCConfig` object defines how to run MOFA on a single HPC system.
The configuration contains two major parts:

#. **Application Details**: Settings for which application to use for each simulation task
   and the proper command to launch them. For example, ``raspa_cmd`` must be the path to a
   RASPA binary for the version defined in ``raspa_version``.

#. **Workflow Configuration**: The :meth:`mofa.hpc.config.HPCConfig.make_parsl_config` generates
   the :class:`~parsl.config.Config` object used by Parsl to launch workers. The configuration
   object contains the Parsl Executors defined for each type of task (e.g., the DFT tasks will run on the
   ``dft_executors`` executors).
   The resources devoted to each tasks are controlled using the ``ai_fraction`` and ``dft_fraction``.

Porting MOFA to a new system involves subclassing :class:`~mofa.hpc.config.HPCConfig`
or :class:`~mofa.hpc.config.SingleJobHPCConfig` and re-defining the methods to create Parsl configurations.
For example, consult the :class:`~mofa.hpc.config.AuroraConfig`
The process for making a new subclass has yet to be streamlined.

The run script reads a configuration function from a Python file.
The file must contain an ``HPCConfig`` object as a variable named ``hpc_config``.
