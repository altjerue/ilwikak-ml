# Scientific ML Projects

Applying machine learning to astrophysical problems where physics domain knowledge drives the modeling decisions. These are not benchmark exercises: each project addresses an open scientific question with public observational data, a physically motivated feature set, and results that are interpretable against a known theoretical framework from past projects.

The connecting thread across all three projects is [Tleco](https://github.com/altjerue/tleco), a Fokker-Planck solver for relativistic outflow modeling. Tleco generates physically labeled synthetic SEDs and light curves across a grid of jet parameters, producing training data that no off-the-shelf dataset can replicate.

---

## Projects

| # | Project | Method | Status |
|---|---------|--------|--------|
| 1 | [Blazar Population ML](#project-1-blazar-population-ml) | Classical ML (PCA, UMAP, GMM, HDBSCAN, GPR) | In progress |
| 2 | [Tleco Neural Emulator](#project-2-tleco-neural-emulator) | PyTorch feedforward / normalizing flow | Planned |
| 3 | [Blazar Light Curve Forecasting](#project-3-blazar-light-curve-forecasting) | LSTM / TCN + foundation model fine-tuning | Planned |

Projects are sequential. Each depends on results from the previous one.

---

## Project 1: Blazar Population ML

**Notebook:** [`blazar_population_ml.ipynb`](./blazar_population_ml.ipynb)

**Reference:** Rueda-Becerril, Harrison & Giannios (2021, MNRAS 501, 4092)

### Scientific Questions

**Primary:** Given multi-wavelength observational features from the 4LAC-DR3 catalog, can a ML model recover the constraints on jet baryon loading ($50 \leq \mu \leq 80$) derived from forward simulation in the 2021 paper? Agreement is independent validation of that result. Disagreement is also scientifically informative.

**Secondary:** Is there a continuous blazar sequence, or are BL Lacs and FSRQs genuinely distinct populations? Can unsupervised clustering on observables recover known subclasses or reveal intermediate objects (BCU)?

### Two-Part Analysis Structure

The project runs on two samples with different feature sets. They are not the same analysis at different scales.

**Part 1: full catalog (3,407 sources)**
Catalog-native features only, no kinematics. The small viewing-angle approximation (Doppler boosting selection enforces $\theta < 1 / \Gamma$ at the population level) justifies using SED features as implicit kinematic proxies. Methods: PCA, UMAP, GMM, HDBSCAN.

**Part 2: MOJAVE cross-matched subsample (334 sources)**
Adds $\beta_{\text{app}}$ and $\Gamma_{\text{min}} = \sqrt{1 + \beta_{\text{app}}^2}$ from VLBI monitoring. The viewing angle affects the observed inverse Compton component independently of $\Gamma_{\text{bulk}}$ (Rueda-Becerril 2014), so both features must enter the model. This subsample is radio-bright and biased toward FSRQs and LSP sources — results are not directly comparable to Part 1 without correcting for MOJAVE selection.

### Features

| Feature | Derivation | Notes |
|---------|-----------|-------|
| `pl_index` | `PL_Index` (native) | Photon spectral index |
| `lp_alpha`, `lp_beta` | `LP_Index`, `LP_beta` (native) | Log-parabola shape |
| `log_nu_syn` | $\log_{10}$(`nu_syn`) | Synchrotron peak frequency; 0-sentinel treated as NaN (777 sources) |
| `log_nuFnu_syn` | $\log_{10}$(`nuFnu_syn`) | Synchrotron peak flux |
| `log_compton_dom` | $\log_{10}$(`HE_nuFnuPeak` $\times 1.602 \times 10^{-6} /$ `nuFnu_syn`) | $\text{MeV cm}^{-2}\text{ s}^{-1} \rightarrow \text{erg cm}^{-2}\text{ s}^{-1}$ conversion required |
| `log_gamma_lum` | $\log_{10}$($4 \pi d_L^2 \times$ `Energy_Flux100`) | $d_L$ from flat $\Lambda\text{CDM}\; (H_{0} = 70, \Omega_m = 0.3)$; `redshift`=0 treated as NaN |
| `var_index` | `Variability_Index` (native) | |
| `beta_app` | `betaMax` from MOJAVE-XVII | Part 2 only; 71 sources have `betaMax` = 0 (no detected superluminal motion) |
| `gamma_min` | $\sqrt{1 + \beta_{\text{app}}^2}$ | Part 2 only; conservative Γ_bulk lower bound |

### Data Acquisition

The `data/` folder is not committed to the repository. Download the following files and place them in `data/` before running the notebook.

**4LAC-DR3** (Ajello et al. 2022, [arXiv:2209.12070](https://arxiv.org/abs/2209.12070)):

Download from the Fermi Science Support Center:
```
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR3/
```
Files needed:
- `table-4LAC-DR3-h.fits` — high Galactic latitude sources, $|b| > 10^{\circ}$, 3,407 sources (primary catalog)
- `table-4LAC-DR3-l.fits` — low Galactic latitude sources (excluded from main analysis, kept for reference)

Alternatively, the VizieR mirror (catalog J/ApJS/263/24) provides the same tables:
```
https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/263/24
```

**MOJAVE-XVII** (Lister et al. 2021, ApJS 255, 30):

Download from VizieR (catalog J/ApJS/255/30), table "list" for HDU1 (BL Lac objects) and table "sample" for HDU2 (quasars/FSRQs):
```
https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/255/30
```
File needed: save as `VizieR-MOJAVE-XVII.fit`

---

## Project 2: Tleco Neural Emulator

**Status:** Not started. Begins after Project 1 is in a presentable state.

### Scientific Question

Can a neural network learn the mapping from physical jet parameters to observable SEDs accurately enough to replace Tleco in parameter inference pipelines? What are the failure modes, and where does the emulator break down?

### Motivation

Tleco integrates the Fokker-Planck equation numerically. It is physically rigorous but computationally expensive across large parameter spaces. A neural surrogate mapping input parameters ($\mu, \sigma, \Gamma, \dot{m}, f_{\text{rec}}$) to SED flux vectors would enable fast parameter space exploration, serve as a differentiable forward model for gradient-based inference, and be useful beyond this specific project.

This is neural emulation of physical simulators, an established approach in climate modeling, cosmological simulations, and plasma physics codes.

---

## Project 3: Blazar Light Curve Forecasting

**Status:** Not started. Begins after Project 2 is in a presentable state. Runs in two phases.

### Scientific Question

Can a general-purpose time series foundation model, fine-tuned with Tleco-generated synthetic light curves, forecast blazar gamma-ray flares? Where does it succeed, where does it fail, and what does the latent structure it learns correspond to physically?

### Key Asset

Tleco integrates the Fokker-Planck equation forward in time. The particle energy distribution evolves at each time step, and the sequence of spectral snapshots is the light curve. Synthetic light curves with physical parameter labels are a direct output of Tleco across the same parameter grid used in Project 2. No competing time series ML paper on blazar variability has labeled synthetic training data from a first-principles physics code.

---

## References

- Ajello et al. 2022 ApJS 263 24
- Davis, Rueda-Becerril & Giannios 2022 MNRAS 513 5766
- Davis, Rueda-Becerril & Giannios 2024 ApJ 976 182
- Lister et al. 2021 ApJS 255 30
- Rueda-Becerril, Harrison & Giannios 2021 MNRAS 501 4092
- Rueda-Becerril, Mimica & Aloy 2017 MNRAS 468 1169
- Rueda-Becerril, Mimica & Aloy 2014 MNRAS 438 1856
