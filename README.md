# NeuralCMS

NeuralCMS is a machine-learning model to compute the gravitational moments and mass of Jupiter given seven chosen parameters setting its interior model. The model is trained on over a million interior model solutions computed with the accurate but computationally demanding concentric Maclaurin spheroid method (CMS; Hubbard 2013 DOI:[10.1088/0004-637X/768/1/43](https://ui.adsabs.harvard.edu/link_gateway/2013ApJ...768...43H/doi:10.1088/0004-637X/768/1/43)). 

NeuralCMS receives the following interior features as input: protosolar helium abundance (setting the overall planetary abundance) $Y_{\rm proto}$, temperature at 1 bar $T_{\rm 1 bar}$, atmospheric heavy materials (anything heavier than helium) abundance $Z_1$, transition pressure between the inner and the outer envelopes $P_{12}$, dilute core extent $m_{\rm dilute}$, dilute core maximum heavy materials abundance $Z_{\rm dilute}$, and compact core normalize radius $r_{\rm core}$, and computes the lower even degree gravity moments and mass.

Here, we share the trained models presented in Ziv et al. 2024, which was accepted for publication in A&A (DOI:[10.1051/0004-6361/202450223](https://doi.org/10.1051/0004-6361/202450223)), together with a Python notebook to load the models, compute a single interior model, and perform a grid search for interior models consistent with Nasa's Juno mission measured gravity moments and mass.

## Installation using pip

This project uses PyTorch, which requires Python 3.8 or higher.

### The required packages:
- python>=3.8
- torch
- numpy
- tqdm
- itertools
- jupyter

Install the requirements:
```
pip install -r requirements.txt
```

## Getting started

Start working with the NeuralCMS in `NeuralCMS_notebook.ipynb`.

## Acknowledgements

Our numerical interior model (CMS), which NeuralCMS was trained on its results, is based on the model from https://github.com/nmovshov/CMS-planet.
