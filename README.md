# hari-plotter

![Logo](res/logo_text.png)

## About

`hari-plotter` enables post-processing, plotting and visualizations of Seldon simulations and output (under development).

## Quick Start

You can use a `micromamba` environment to get started, and get all your dependencies (or choose your own poison!):

```bash
micromamba create -f environment.yml # Just once 
micromamba activate hairplotterenv # To activate, every time 
```

If you prefer `pip`:

```
pip install .
```

or

```
pip install .[qt]
```

if you whant to use the GUI

## Test

You can use `pytest` and `covarage` for tests:

```bash
coverage run --source=hari_plotter -m pytest tests/ # To test
coverage report # To get the report
```
