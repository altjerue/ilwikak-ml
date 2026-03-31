from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

warnings.filterwarnings("ignore")

# Selecting cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def safe_float(col):
    return np.ma.filled(col.data, np.nan).astype(float)


def safe_str(col):
    return np.array([str(v).strip() if v is not np.ma.masked else "" for v in col])


def safe_str_lower(col):
    return np.array(
        [str(v).strip().lower() if v is not np.ma.masked else "" for v in col]
    )


def lac_features(lac_sub):
    z = safe_float(lac_sub["Redshift"])
    z = np.where(z <= 0, np.nan, z)
    nu = safe_float(lac_sub["nu_syn"])
    nu = np.where(nu <= 0, np.nan, nu)
    nf = safe_float(lac_sub["nuFnu_syn"])
    nf = np.where(nf <= 0, np.nan, nf)
    hp = safe_float(lac_sub["HE_EPeak"])
    hn = safe_float(lac_sub["HE_nuFnuPeak"])
    hn = np.where(hn <= 0, np.nan, hn)
    he_erg = (hn * u.MeV / u.cm**2 / u.s).to(u.erg / u.cm**2 / u.s).value
    cd = np.where((~np.isnan(he_erg)) & (~np.isnan(nf)) & (nf > 0), he_erg / nf, np.nan)
    ef = safe_float(lac_sub["Energy_Flux100"])
    vi = safe_float(lac_sub["Variability_Index"])
    fv = safe_float(lac_sub["Frac_Variability"])
    fv = np.where(fv <= 0, np.nan, fv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # Calculating luminosity distance
    dl = np.where(
        ~np.isnan(z),
        cosmo.luminosity_distance(np.where(~np.isnan(z), z, 0.1)).to(u.cm).value,
        np.nan,
    )
    lum = np.where(~np.isnan(z), 4.0 * np.pi * dl**2 * ef, np.nan)

    return dict(
        class_=safe_str(lac_sub["CLASS"]),
        sed_class=safe_str(lac_sub["SED_class"]),
        redshift=z,
        pl_index=safe_float(lac_sub["PL_Index"]),
        lp_alpha=safe_float(lac_sub["LP_Index"]),
        lp_beta=safe_float(lac_sub["LP_beta"]),
        log_nu_syn=np.log10(np.where(nu > 0, nu, np.nan)),
        log_nuFnu_syn=np.log10(np.where(nf > 0, nf, np.nan)),
        log_he_epeak=np.log10(np.where(hp > 0, hp, np.nan)),
        log_compton_dom=np.log10(np.where(cd > 0, cd, np.nan)),
        log_gamma_lum=np.log10(np.where(lum > 0, lum, np.nan)),
        var_index=vi,
        frac_var=fv,
    )


DATA_DIR = Path("../data")
LAC_H = DATA_DIR / "table-4LAC-DR3-h.fits"
MOJAVE = DATA_DIR / "VizieR-MOJAVE-XVII.fit"

lac = Table.read(LAC_H)
lac_feat = lac_features(lac)
df1 = pd.DataFrame(
    {
        "source_name": [str(v).strip() for v in lac["Source_Name"]],
        "ra": safe_float(lac["RAJ2000"]),
        "dec": safe_float(lac["DEJ2000"]),
        "class": lac_feat["class_"],
        "sed_class": lac_feat["sed_class"],
        **{k: v for k, v in lac_feat.items() if k not in ["class_", "sed_class"]},
    }
)
df1.head(10)
