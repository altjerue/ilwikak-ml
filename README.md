# Scientific ML Projects

Applying machine learning to astrophysical problems where physics domain knowledge drives the modeling decisions. These are not benchmark exercises: each project addresses an open scientific question with public observational data, a physically motivated feature set, and results that are interpretable against a known theoretical framework.

The connecting thread across all three projects is [Tleco](https://github.com/altjerue/tleco), an in-house Fokker-Planck solver for blazar emission modeling. Tleco generates physically labeled synthetic SEDs and light curves across a grid of jet parameters, producing training data that no off-the-shelf dataset can replicate. That is the differentiator.

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

**Primary:** Given multi-wavelength observational features from the 4LAC-DR3 catalog, can a ML model recover the constraints on jet baryon loading (50 ≤ μ ≤ 80) derived from forward simulation in the 2021 paper? Agreement is independent validation of that result. Disagreement is also scientifically informative.

**Secondary (sets up the primary):** Is there a continuous blazar sequence, or are BL Lacs and FSRQs genuinely distinct populations? Can unsupervised clustering on observables recover known subclasses or reveal intermediate objects (BCU)?

### Two-Part Analysis Structure

The project runs on two samples with different feature sets. They are not the same analysis at different scales.

**Part 1 — full catalog (3,407 sources):** Catalog-native features only, no kinematics. The small viewing-angle approximation (Doppler boosting selection enforces θ < 1/Γ at the population level) justifies using SED features as implicit kinematic proxies. Methods: PCA, UMAP, GMM, HDBSCAN.

**Part 2 — MOJAVE cross-matched subsample (334 sources):** Adds β_app and Γ_min = √(1 + β_app²) from VLBI monitoring. The viewing angle affects the observed inverse Compton component independently of Γ_bulk (Rueda-Becerril 2014), so both features must enter the model. This subsample is radio-bright and biased toward FSRQs and LSP sources — results are not directly comparable to Part 1 without correcting for MOJAVE selection.

### Features

| Feature | Derivation | Notes |
|---------|-----------|-------|
| `pl_index` | `PL_Index` (native) | Photon spectral index |
| `lp_alpha`, `lp_beta` | `LP_Index`, `LP_beta` (native) | Log-parabola shape |
| `log_nu_syn` | $\log_{10}($\nu_{\text{syn}}) | Synchrotron peak frequency; 0-sentinel treated as NaN (777 sources) |
| `log_nuFnu_syn` | log₁₀(`nuFnu_syn`) | Synchrotron peak flux |
| `log_compton_dom` | log₁₀(`HE_nuFnuPeak` × 1.602e-6 / `nuFnu_syn`) | MeV cm⁻² s⁻¹ → erg cm⁻² s⁻¹ conversion required |
| `log_gamma_lum` | log₁₀(4π d_L² × `Energy_Flux100`) | d_L from flat ΛCDM (H₀=70, Ω_m=0.3); redshift=0 treated as NaN |
| `var_index` | `Variability_Index` (native) | |
| `beta_app` | `betaMax` from MOJAVE-XVII | Part 2 only; 71 sources have betaMax=0 (no detected superluminal motion) |
| `gamma_min` | √(1 + $\beta_{\text{app}}^2) | Part 2 only; conservative Γ_bulk lower bound |

### Missingness Summary

| Feature | Part 1 (N=3,407) | Part 2 (N=334) |
|---------|-----------------|----------------|
| `redshift` | 47.0% | 9.9% |
| `log_nu_syn` | 22.8% | 2.1% |
| `log_compton_dom` | 34.0% | 6.6% |
| `log_gamma_lum` | 47.0% | 9.9% |
| `frac_var` | 25.7% | 4.2% |

Part 2 has lower missingness because MOJAVE selects bright radio sources with better multiwavelength coverage. This is further evidence of selection bias, not better data quality in the sample.

### Methods

- Dimensionality reduction: PCA, UMAP
- Clustering: Gaussian Mixture Models (k=2–5, BIC/AIC selection), HDBSCAN on UMAP embedding
- Regression (μ recovery): Random Forest (baseline), Gaussian Process Regression (preferred at this sample size), XGBoost
- Uncertainty quantification: Bayesian approaches, conformal prediction, bootstrap ensembles

Classical ML is appropriate here. The dataset is at most a few thousand points. Deep learning is underpowered at this scale and harder to interpret — interpretability is required to make a scientific claim.

### Data Acquisition

The `data/` folder is not committed to the repository. Download the following files and place them in `Science/data/` before running the notebook.

**4LAC-DR3** (Ajello et al. 2022, arXiv:2209.12070):

Download from the Fermi Science Support Center:
```
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR3/
```
Files needed:
- `table-4LAC-DR3-h.fits` — high Galactic latitude sources, |b| > 10°, 3,407 sources (primary catalog)
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

**Derived parquet files** (generated by the notebook, no download needed):
- `data/part1_4lac_dr3.parquet` — 3,407 sources, 16 features
- `data/part2_4lac_mojave.parquet` — 334 sources, 19 features (adds β_app, e_β_app, γ_min)

---

## Project 2: Tleco Neural Emulator

**Status:** Not started. Begins after Project 1 is in a presentable state.

### Scientific Question

Can a neural network learn the mapping from physical jet parameters to observable SEDs accurately enough to replace Tleco in parameter inference pipelines? What are the failure modes, and where does the emulator break down?

### Motivation

Tleco integrates the Fokker-Planck equation numerically. It is physically rigorous but computationally expensive across large parameter spaces. A neural surrogate mapping input parameters (μ, σ, Γ, ṁ, f_rec) to SED flux vectors would enable fast parameter space exploration, serve as a differentiable forward model for gradient-based inference, and be useful beyond this specific project.

This is neural emulation of physical simulators, an established approach in climate modeling, cosmological simulations, and plasma physics codes.

### Approach

1. Generate training data by running Tleco over a Latin hypercube or Sobol sequence in (μ, σ, Γ, ṁ, f_rec)
2. Train a feedforward network or normalizing flow to predict SED flux vectors at fixed frequency bins
3. Validate against held-out Tleco runs; characterize failure modes and coverage gaps
4. Integrate the emulator as a forward model in a Bayesian inference loop (MCMC or nested sampling)
5. Fit observed SEDs from 4LAC-DR3; compare recovered parameters to the 2021 results

### Tools

PyTorch · NumPyro or PyMC · MLflow or Weights and Biases · Tleco

### Data Acquisition

No external downloads required. Training data is generated by running Tleco over the parameter grid. Grid bounds are set from the 2021 paper (μ: 50–80, σ: 0.1–1.5, Γ: 5–30, ṁ: 0.01–0.5).

---

## Project 3: Blazar Light Curve Forecasting

**Status:** Not started. Begins after Project 2 is in a presentable state. Runs in two phases.

### Scientific Question

Can a general-purpose time series foundation model, fine-tuned with Tleco-generated synthetic light curves, forecast blazar gamma-ray flares? Where does it succeed, where does it fail, and what does the latent structure it learns correspond to physically?

### Key Asset

Tleco integrates the Fokker-Planck equation forward in time. The particle energy distribution evolves at each time step, and the sequence of spectral snapshots is the light curve. Synthetic light curves with physical parameter labels are a direct output of Tleco across the same parameter grid used in Project 2. No competing time series ML paper on blazar variability has labeled synthetic training data from a first-principles physics code.

### Phase 3.1 — Baseline Forecasting

Given N days of gamma-ray flux history, predict the next M days. Baselines: ARIMA, Prophet. Models: LSTM, Temporal Convolutional Network. Tleco synthetic curves used for data augmentation where observed coverage is sparse. Source selection guided by the 4LAC-DR3 variability index.

**Tools:** PyTorch · statsmodels · Fermi-LAT data server

### Phase 3.2 — Foundation Model Fine-Tuning

Zero-shot evaluation of a pre-trained time series foundation model (Chronos, TimesFM, or Moirai) on blazar light curves, followed by fine-tuning using Tleco-simulated curves for domain adaptation. Probe the learned latent representations: do sources cluster by physical class? Do latent dimensions correlate with parameters recovered in Project 1?

**Tools:** HuggingFace Transformers · PyTorch · Fermi-LAT data server · Tleco

### Data Acquisition

**Fermi-LAT light curves** (observed data):

Download from the Fermi-LAT Light Curve Repository:
```
https://fermi.gsfc.nasa.gov/ssc/data/access/lat/LightCurveRepository/
```
Select sources by 4LAC-DR3 name; choose weekly or monthly binning. The repository provides per-source ASCII tables with flux, flux uncertainty, and upper limits.

**Synthetic light curves:** Generated by Tleco (no external download).
