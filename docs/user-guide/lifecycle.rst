Lifecycle
=========

The information about a candidate MOF is accrued in a :class:`~mofa.model.MOFRecord` object.
The record contains identifying information about a MOF that remains unchanged during a run,
and a series of properties which are gradually gathered during execution.


.. contents:: Record Components
    :depth: 2

MOF Definition
--------------

MOFA starts by defining a MOF uniquely using the type of metal node,
which molecules connect them, and the connection topology.
Each portion is defined with a different data model.

- *Metal Node*: The XYZ coordinates and name described by a :class:`~mofa.model.NodeDescription` object.
- *Ligand*: Each ligand is a molecule described using a :class:`~mofa.model.LigandDescription` object, which
  holds a molecular graph description (i.e., SMILES string) and 3D coordinates of the atoms.
  The description also retains data about how it was created (e.g., the atoms used to prompt a generator, model version)
  and how it links to the larger MOF (e.g., identity of the anchor atoms shared with the node).

The node and ligand definition are included in the :class:`~mofa.model.MOFRecord` along with the 3D topology
and catenation degree.
The node, ligand, topology, and catenation uniquely define a MOF.

The MOF is also assigned by a globally-unique name and can be further identified with other names,
such as its record identifier from another database.

Assembly
--------

Most computations of a MOF require defining a 3D structure, which MOFA generates using the :meth:`~mofa.assembly.assemble.assemble_mof` function.

The structure is stored in a `common text-based format <https://www.vasp.at/wiki/index.php/POSCAR>`_
in the MOF record.

The as-assembled structure is an estimate that maybe be further refined by MOFA.

Property Calculation
--------------------

Each type of property calculation performed by MOFs has at least one attribute in the :class:`~mofa.model.MOFRecord`.

The property attributes are nested dictionaries where the first key is a name signifying the level of fidelity
and any metadata about the computation.

The steering process is responsible for storing data with the appropriate name.

.. note::

    TODO: Describe the types of properties and the associated codes in greater detail.

Workflow Tracking
-----------------

The ``times`` and ``in_progress`` fields are used in MOFA to track when different data were acquired and
if any computations are on-going.
