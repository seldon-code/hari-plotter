.. _installation:

Installation
============

This section covers the installation instructions for `hari-plotter`.

.. note::

    Ensure you have the necessary permissions to install software on your machine.

Quick Start
-----------

### Using Micromamba

You can use a `micromamba` environment to get started and install all your dependencies:

.. code-block:: bash

    micromamba create -f environment.yml  # Run this command only once to create the environment
    micromamba activate hairplotterenv  # Run this command every time to activate the environment

### Using pip

If you prefer using `pip`, you can install `hari-plotter` with:

.. code-block:: bash

    pip install .

For using the Graphical User Interface (GUI), install with the `qt` extra:

.. code-block:: bash

    pip install .[qt]

Testing the Installation
------------------------

You can use `pytest` and `coverage` to run tests and check the coverage of your code:

.. code-block:: bash

    coverage run --source=hari_plotter -m pytest tests/  # Run tests
    coverage report  # Get the coverage report
