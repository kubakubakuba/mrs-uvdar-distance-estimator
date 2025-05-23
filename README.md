# Dataset (old)

[old: dataset link, requires a CTU account](https://campuscvut-my.sharepoint.com/:f:/g/personal/pelcjaku_cvut_cz/En5PCJdR1GBIky51O9Q46fQBw8nHX435BbOwW9Sc2sZf7g?e=bdEQoG)

# Distance / frequency / incidence angle analysis

The main file for this analysis is the `main.ipynb` jupyter notebook. The preprocessed dataset files (precomputed event averages) are in the `data/` directory stored in `.npz` files.

# PnP

The dataset used for PnP is stored in `pnp_dataset/` in `.toml` or `.json` files (with prelabeled LED locations). The files for processing are called `pnp*.py`, with the `PnPSolver.py` being the
wrapper around the solving process.