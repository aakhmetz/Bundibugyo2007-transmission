# %% Machinery
exec(open("../../../psw/psw.py").read())

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 90
import matplotlib.pyplot as plt
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams.update({"figure.constrained_layout.use": True})
import os
import pathlib
import platform

plt.rcParams["font.family"] = "sans-serif"

import numpy as np
import pandas as pd
import seaborn as sns
from termcolor import colored

mainstandirname = "../../../Taiwan_Backup/Taiwan-Bundibugyo_2026"
os.makedirs(mainstandirname, exist_ok=True)

figuresdir = "../figures"
os.makedirs(figuresdir, exist_ok=True)

import warnings

import arviz as az
import cmdstanpy as cmdstan

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="arviz")
standistribdir = "../../../../CmdStan"
cmdstan.set_cmdstan_path(standistribdir)

func_dict = {
    "q2.5": lambda x: np.percentile(x, 2.5),
    "q5": lambda x: np.percentile(x, 5),
    "q25": lambda x: np.percentile(x, 25),
    "median": lambda x: np.percentile(x, 50),
    "q75": lambda x: np.percentile(x, 75),
    "q90": lambda x: np.percentile(x, 90),
    "q95": lambda x: np.percentile(x, 95),
    "q97.5": lambda x: np.percentile(x, 97.5),
}


def get_stats(cmdstan_data, varnames, ignore_nan_=True, round_to_=5):
    # Ensure varnames is a list
    if isinstance(varnames, str):
        varnames = [varnames]

    # mean, sd, diagnostics and 95% HDI (column names differ across ArviZ versions)
    stats_95 = (
        az.summary(
            cmdstan_data,
            round_to=round_to_,
            var_names=varnames,
            ci_kind="hdi",
            ci_prob=0.95,
            skipna=ignore_nan_,
        )
        .reset_index()
        .rename(columns={"index": "var"})
    )

    hdi95_lb_col = [
        c for c in stats_95.columns if "hdi" in c and ("lb" in c or "2.5" in c)
    ][0]
    hdi95_ub_col = [
        c for c in stats_95.columns if "hdi" in c and ("ub" in c or "97.5" in c)
    ][0]

    stats_95 = stats_95.rename(
        columns={hdi95_lb_col: "hdi2.5", hdi95_ub_col: "hdi97.5"}
    )

    stats = stats_95.loc[
        :, ["var", "mean", "sd", "hdi2.5", "hdi97.5", "ess_bulk", "ess_tail", "r_hat"]
    ]

    # 50% HDI
    stats_50 = (
        az.summary(
            cmdstan_data,
            round_to=round_to_,
            var_names=varnames,
            ci_kind="hdi",
            ci_prob=0.50,
            skipna=ignore_nan_,
        )
        .reset_index()
        .rename(columns={"index": "var"})
    )

    hdi50_lb_col = [
        c for c in stats_50.columns if "hdi" in c and ("lb" in c or "25" in c)
    ][0]
    hdi50_ub_col = [
        c for c in stats_50.columns if "hdi" in c and ("ub" in c or "75" in c)
    ][0]

    stats_50 = stats_50.loc[:, ["var", hdi50_lb_col, hdi50_ub_col]].rename(
        columns={hdi50_lb_col: "hdi25", hdi50_ub_col: "hdi75"}
    )

    # Merge HDIs together
    stats = pd.merge(stats, stats_50, on="var")

    # percentiles
    percentiles_to_calc = {
        "q2.5": 2.5,
        "q5": 5,
        "q25": 25,
        "median": 50,
        "q75": 75,
        "q90": 90,
        "q95": 95,
        "q97.5": 97.5,
        "q99": 99,
    }

    percentile_rows = []
    posterior = cmdstan_data.posterior

    for var in varnames:
        if var in posterior:
            da = posterior[var]

            # reduce over the chain and draw dimensions
            dims_to_reduce = [d for d in da.dims if d not in ("chain", "draw")]

            if not dims_to_reduce:
                # Scalar variable case
                row = {"var": var}
                for name, q in percentiles_to_calc.items():
                    row[name] = np.round(np.nanpercentile(da.values, q), round_to_)
                percentile_rows.append(row)
            else:
                # vector or array: one row per element
                df_var = da.to_dataframe().reset_index()
                groupby_cols = [
                    c for c in df_var.columns if c not in ("chain", "draw", var)
                ]

                if groupby_cols:
                    for keys, group in df_var.groupby(groupby_cols):
                        # ArviZ-style name, e.g. var[0]
                        key_str = ",".join(
                            map(str, keys if isinstance(keys, tuple) else [keys])
                        )
                        row = {"var": f"{var}[{key_str}]"}
                        for name, q in percentiles_to_calc.items():
                            row[name] = np.round(
                                np.nanpercentile(group[var].values, q), round_to_
                            )
                        percentile_rows.append(row)
                else:
                    row = {"var": var}
                    for name, q in percentiles_to_calc.items():
                        row[name] = np.round(
                            np.nanpercentile(df_var[var].values, q), round_to_
                        )
                    percentile_rows.append(row)

    pct_df = pd.DataFrame(percentile_rows)

    stats = pd.merge(pct_df, stats, on="var")

    # split "var[i]" into the variable name and a 1-based time index
    stats["time"] = stats["var"].apply(
        lambda st: st[st.find("[") + 1 : st.find("]")] if "[" in st else "NA"
    )
    stats["time"] = [
        "NA" if y == "NA" else int(x) + 1 for x, y in zip(stats["time"], stats["time"])
    ]
    stats["var"] = stats["var"].apply(
        lambda st: st[: st.find("[")] if "[" in st else st
    )

    final_cols = [
        "var",
        "time",
        "mean",
        "sd",
        "q2.5",
        "q25",
        "median",
        "q75",
        "q97.5",
        "q5",
        "q90",
        "q95",
        "q99",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ]

    return stats.loc[:, final_cols]


def get_stats_2d(cmdstan_data, varnames, rounding=2):
    # include mean and hpd
    stats = (
        az.summary(cmdstan_data, var_names=varnames, hdi_prob=0.95, round_to=rounding)
        .loc[:, ["mean", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "ess_tail", "r_hat"]]
        .reset_index()
        .rename(columns={"index": "var", "hdi_2.5%": "hdi2.5", "hdi_97.5%": "hdi97.5"})
    )
    stats = (
        az.summary(cmdstan_data, var_names=varnames, hdi_prob=0.50, round_to=rounding)
        .loc[:, ["hdi_25%", "hdi_75%"]]
        .reset_index()
        .rename(columns={"index": "var", "hdi_25%": "hdi25", "hdi_75%": "hdi75"})
        .merge(stats, left_on="var", right_on="var")
    )
    # include percentiles
    stats = (
        az.summary(
            cmdstan_data,
            var_names=varnames,
            stat_funcs=func_dict,
            extend=False,
            round_to=rounding,
        )
        .reset_index()
        .rename(columns={"index": "var"})
        .merge(stats, left_on="var", right_on="var")
    )
    stats["time"] = stats["var"].apply(lambda st: st[st.find("[") + 1 : st.find("]")])
    stats["time"] = [
        "NA" if "[" not in y else x for x, y in zip(stats["time"], stats["var"])
    ]
    stats["var"] = stats["var"].apply(
        lambda st: st[: st.find("[")] if "[" in st else st
    )
    return stats.loc[
        :,
        [
            "var",
            "time",
            "mean",
            "hdi2.5",
            "hdi25",
            "hdi75",
            "hdi97.5",
            "q2.5",
            "q25",
            "median",
            "q75",
            "q97.5",
            "ess_bulk",
            "ess_tail",
            "r_hat",
        ],
    ]


from IPython.display import HTML


def show_stats(stats, sig=None):
    """VSCode-like table: nowrap cells + horizontal scroll. No jinja2 needed."""
    out = stats.round(sig) if sig is not None else stats
    html = out.to_html(index=False, border=0)
    html = html.replace(
        "<table", '<table style="white-space:nowrap; text-align:right;"'
    )
    return HTML(f'<div style="overflow-x:auto; max-width:100%;">{html}</div>')


def write_bash_file(
    execfile,
    standirname,
    max_depth_=13,
    num_chains=5,
    num_samples=12500,
    num_warmup=2500,
    num_threads=4,
):
    stanscriptdir = "../Dropbox/" + standirname[9:]
    str_ = f"""#!/bin/bash
cwd=$(pwd)
cd {standistribdir}
STAN_THREADS=true make -j4 {stanscriptdir}/{execfile}
cd {stanscriptdir}
seeds=(20260513 100 200 20260524 20260509 20260525 20260526 20260527)
declare -a seeds
for i in {{1..{num_chains}}}
do
    echo Running ${{i}}
    SEEDNUMBER=${{seeds[$i-1]}}
    ./{execfile} \\
        method=sample num_samples={num_samples} num_warmup={num_warmup} thin=1 save_warmup=0 adapt delta=0.99 \\
            algorithm=hmc \\
                engine=nuts max_depth={max_depth_} \\
        random seed=${{SEEDNUMBER}} \\
        id=$i \\
        data file=Data.json \\
        init=Inits.json \\
        num_threads={num_threads} \\
        output file=trace-$i.csv \\
            diagnostic_file=diagnostics-$i.csv > output-$i.txt &
done
echo Finished
"""
    with open(os.path.join(standirname, "fit_bash.sh"), "w+") as f:
        f.write(str_)
    return True


#

# %% Loading data: Weekly cases of Ebola virus in the 2007-2008 outbreak in Bundibugyo, Uganda (Wamala et al., 2010)
df = pd.read_csv("../data/wamala2010_weekly_cases.csv")
# add total column
df["count_total"] = df["count_fatal"] + df["count_non_fatal"]
min_week = df["week"].min()
print("Min week: ", min_week)

# Add week_index column
df["week_index"] = df["week"] - min_week + 1

# Outbreak statistics
print(
    f"Total weeks: {len(df)} (from week {df['week'].min()} to week {df['week'].max()})"
)
print(f"Total fatal cases: {df['count_fatal'].sum()}")
print(f"Total non-fatal cases: {df['count_non_fatal'].sum()}")
print(f"Total cases: {df['count_fatal'].sum() + df['count_non_fatal'].sum()}")

df[:5]
#

# %% Intervention time
intervention_effect_week = 48  # week 48, start of control measures
changepoint = df.loc[df["week"] == intervention_effect_week, "week_index"].values[0]

print(changepoint)
#

# %% Epicurve of weekly cases
plt.figure(figsize=(6, 4.6))
plt.bar(df["week"], df["count_fatal"], label="Fatal", color="white", edgecolor="black")
plt.bar(
    df["week"],
    df["count_non_fatal"],
    bottom=df["count_fatal"],
    label="Non-fatal",
    color="black",
    edgecolor="black",
)
plt.xlabel("Week in 2007")
plt.ylabel("New cases")
plt.xticks(df["week"])
plt.tick_params(axis="x", which="both", bottom=False, top=False)
plt.axvline(
    intervention_effect_week - 0.5,
    color="red",
    linestyle="--",
    label="Intervention effect",
)

plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
#


# %% Executing stan model
import glob
import shutil
import subprocess

pathfinder_ = True

from scipy.stats import gamma as _gam
from scipy.stats import triang

W = df.shape[0]
cases_fatal = df["count_fatal"].astype("int").tolist()
cases_non_fatal = df["count_non_fatal"].astype("int").tolist()
t_int = float(changepoint)

tau_rep_values = [2, 4, 8]  # tau_rep values of the ascertainment-ramp sensitivity analysis


# daily-discretised triangular incubation period and gamma TOST for the daily_triangular models
_Finc = triang(c=(7 - 2) / (20 - 2), loc=2, scale=18)  # incubation Triangular(2, 7, 20)
_Ftost = _gam(a=(4.8 / 2.4) ** 2, scale=(2.4**2) / 4.8)  # TOST gamma(mean 4.8, sd 2.4)
d_inc_max, d_ost_max = 28, 21
_s = np.arange(1, d_inc_max + 1)
f_inc_daily = _Finc.cdf(_s + 0.5) - _Finc.cdf(_s - 0.5)
f_inc_daily = f_inc_daily / f_inc_daily.sum()
_y = np.arange(0, d_ost_max + 1)
f_tost_daily = _Ftost.cdf(_y + 0.5) - np.where(_y > 0, _Ftost.cdf(_y - 0.5), 0.0)
f_tost_daily = f_tost_daily / f_tost_daily.sum()

_common = dict(
    W=W, cases_fatal=cases_fatal, cases_non_fatal=cases_non_fatal, t_int=t_int
)


def _data_weekly(
    tau=None,
):  # tau=None -> step model; tau set -> ramp model (sensitivity)
    d = {
        **_common,
        "w_max": 3,
        "f_inc": np.array([0.6044, 0.3682, 0.0274]),
        "f_tost": np.array([0.3341, 0.6406, 0.0250, 0.0003]),
    }
    if tau is not None:
        d["tau_rep"] = float(tau)
    return d


def _data_daily_tri():  # step model (no tau_rep)
    return {
        **_common,
        "d_max": d_inc_max,
        "d_ost_max": d_ost_max,
        "f_inc": f_inc_daily,
        "f_tost": f_tost_daily,
    }


def _data_daily_gamma(
    mean_inc, sd_inc, tau=None
):  # gamma AND lognormal (both parameterised by mean + sd)
    d = {
        **_common,
        "d_inc_max": d_inc_max,
        "d_ost_max": d_ost_max,
        "mean_inc": float(mean_inc),
        "sd_inc": float(sd_inc),
        "mean_ost": 4.8,
        "sd_ost": 2.4,
    }
    if tau is not None:
        d["tau_rep"] = float(tau)  # ascertainment-ramp variant
    return d


def _data_daily_weibull(
    mean_inc, shape
):  # Weibull incubation; shape from get.weibull.par
    return {
        **_common,
        "d_inc_max": d_inc_max,
        "d_ost_max": d_ost_max,
        "mean_inc": float(mean_inc),
        "param1_Weibull_inc": float(shape),
        "mean_ost": 4.8,
        "sd_ost": 2.4,
    }


_SRC = "stan_src/Rt_Bundibugyo-"

# Incubation kernels fit to the SAME quantiles (q = 2,4,7,10,20 d) via rriskDistributions:
#   gamma     get.gamma.par   -> mean 7.71, sd 4.68
#   Weibull   get.weibull.par -> mean 7.48, shape 1.857
#   lognormal get.lnorm.par   -> mean 8.19, sd 6.01  (Stan recomputes meanlog/sdlog from mean+sd)
_daily_incub = [
    ("triangular", "daily_triangular_step_", _data_daily_tri()),
    ("gamma_BVD", "daily_gamma_step_", _data_daily_gamma(7.71, 4.68)),
    ("gamma_EVD", "daily_gamma_step_", _data_daily_gamma(11.4, 5.20)),
    ("Weibull_BVD", "daily_Weibull_step_", _data_daily_weibull(7.4754, 1.8568)),
    ("lognormal_BVD", "daily_lognormal_step_", _data_daily_gamma(8.1937, 6.0145)),
]
_daily_mech = [  # stem only; CFR suffix (_gamma_approx / _quasi-Binomial) added in _cfr()
    ("semi_uniform", "semi_mechanistic_uniform"),
    ("semi_dirichlet", "semi_mechanistic_dirichlet"),
    ("fully_vtm", "latent_normal_vtm"),
    ("fully_cv", "latent_normal_cv"),
]

# base grid, indexed by basename -> (basename, stem_without_cfr_suffix, data)
_grid_list = [
    (
        "Rt_weekly_triangular_step",
        _SRC + "weekly_triangular_step_semi_mechanistic",
        _data_weekly(),
    ),
    (
        "Rt_weekly_triangular_fully_vtm",
        _SRC + "weekly_triangular_step_latent_normal_vtm",
        _data_weekly(),
    ),
    (
        "Rt_weekly_triangular_fully_cv",
        _SRC + "weekly_triangular_step_latent_normal_cv",
        _data_weekly(),
    ),
]
for _inc, _pref, _dat in _daily_incub:
    for _mlab, _mstem in _daily_mech:
        _grid_list.append((f"Rt_daily_{_inc}_{_mlab}", _SRC + _pref + _mstem, _dat))
_grid = {b: (b, s, d) for b, s, d in _grid_list}


def _cfr(basename, cfr_variants=(True, False)):
    """Expand a grid basename over CFR likelihood(s). QB -> '_quasiBinomial' tag +
    _quasi-Binomial.stan; gamma-approx -> bare basename + _gamma_approx.stan. Skips missing files."""
    out = []
    _, stem, data = _grid[basename]
    for qb in cfr_variants:
        f = stem + ("_quasi-Binomial.stan" if qb else "_gamma_approx.stan")
        if os.path.exists(f):
            out.append((basename + ("_quasiBinomial" if qb else ""), f, data))
        else:
            print(colored(f"skip (missing file): {os.path.basename(f)}", "yellow"))
    return out


# Model set
runs_all = []
# (1) Appendix Tables 1 and 2: daily models, BVD incubation period (gamma, Weibull, lognormal,
#     triangular) x 4 structures, both case-fatality likelihoods
for _inc in ["gamma_BVD", "Weibull_BVD", "lognormal_BVD", "triangular"]:
    for _m in ["semi_uniform", "semi_dirichlet", "fully_vtm", "fully_cv"]:
        runs_all += _cfr(f"Rt_daily_{_inc}_{_m}", (True, False))
# (2) Appendix Table 3: weekly triangular models, quasi-Binomial CFR
for _b in [
    "Rt_weekly_triangular_step",
    "Rt_weekly_triangular_fully_vtm",
    "Rt_weekly_triangular_fully_cv",
]:
    runs_all += _cfr(_b, (True,))
# (3) Appendix Table 4: daily gamma EVD models, quasi-Binomial CFR
for _m in ["semi_uniform", "fully_vtm", "fully_cv"]:
    runs_all += _cfr(f"Rt_daily_gamma_EVD_{_m}", (True,))
# (4) Appendix Figures 2 and 3: ascertainment-ramp models (weekly and daily), tau_rep = 2, 4, 8
runs_all += [
    (f"Rt_weekly_triangular_tau_{t}", _SRC + "weekly_triangular.stan", _data_weekly(t))
    for t in tau_rep_values
]
runs_all += [
    (
        f"Rt_daily_gamma_BVD_tau_{t}",
        _SRC + "daily_gamma_ramp_semi_mechanistic_uniform_quasi-Binomial.stan",
        _data_daily_gamma(7.71, 4.68, tau=t),
    )
    for t in tau_rep_values
]

# Optional substring filter on the basename (None = every model). upload_and_launch() clears
# each model folder before uploading.
_ONLY = None
runs_upload = [r for r in runs_all if _ONLY is None or _ONLY in r[0]]

print(
    colored(
        f"\nUPLOAD SET: {len(runs_upload)} fits"
        + (f"  (filtered from {len(runs_all)} by _ONLY={_ONLY!r})" if _ONLY else ""),
        "cyan",
    )
)
for _bn, _f, _ in runs_upload:
    print(f"  {_bn:52s}  {os.path.basename(_f)}")


def upload_and_launch(runs):
    """For each (basename, stan_src_file, stan_data): stage the model directory locally
    (Stan code, Data.json, Pathfinder Inits.json, fit_bash.sh), upload it, and launch the
    fit on the remote server. Shared by every model set (main grid, tau_rep, eps/theta)."""
    for basename, stan_src_file, stan_data in runs:
        standirname = os.path.join(mainstandirname, basename)
        os.makedirs(standirname, exist_ok=True)
        # clear the model folder
        for fl in glob.glob(os.path.join(standirname, "*")):
            if os.path.isfile(fl):
                os.remove(fl)

        stan_code_file = os.path.join(standirname, f"{basename}.stan")
        shutil.copyfile(stan_src_file, stan_code_file)

        stan_data_file = os.path.join(standirname, "Data.json")
        cmdstan.write_stan_json(stan_data_file, stan_data)

        if pathfinder_ == True:
            model = cmdstan.CmdStanModel(stan_file=stan_code_file)
            try:
                pathfinder_fit = model.pathfinder(
                    data=stan_data_file,
                    seed=20260306,
                    sig_figs=14,
                    num_paths=1,
                    tol_obj=1e-12,
                    tol_grad=1e-12,
                    tol_param=1e-12,
                    history_size=100,
                    max_lbfgs_iters=100,
                )
                inits_data = pathfinder_fit.create_inits()[-1]
                print(inits_data)
            except Exception as _e:
                # Pathfinder can fail on a non-finite gradient; fall back to default inits
                print(
                    colored(
                        f"pathfinder failed for {basename}; using default inits",
                        "yellow",
                    )
                )
                print(_e)
                inits_data = {"log_R": [-0.5, 0.5]}
        else:
            inits_data = dict(
                {
                    "log_R": [-0.5, 0.5],
                }
            )

        # writing initial values to a file
        stan_inits_file = os.path.join(standirname, "Inits.json")
        cmdstan.write_stan_json(stan_inits_file, inits_data)

        # writing bash file to run the model
        write_bash_file(basename, standirname, max_depth_=12)

        fls = [
            f
            for f in os.listdir(standirname)
            if os.path.isfile(os.path.join(standirname, f))
        ]
        for fl in fls:
            print(fl)
            local_path = os.path.join(standirname, fl)
            remote_path = (
                f"sftp://{remote_IP}/home/andrei/Dropbox/{standirname[9:]}/{fl}"
            )
            subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--insecure",
                    "--ftp-create-dirs",
                    "--user",
                    f"{remote_userID}:{remote_userPSW}",
                    "-T",
                    local_path,
                    remote_path,
                ],
                check=False,
            )

        subprocess.run(
            f"sshpass -p {remote_userPSW} ssh {remote_userID}@{remote_IP} -n -f "
            f"\"sh -c 'export LC_ALL=C LANG=C; cd Dropbox/{standirname[9:]}; rm {basename}; "
            f"bash --login -s < fit_bash.sh' >/dev/null 2>&1 </dev/null &\"",
            shell=True,
            check=False,
        )


# set to True to stage, upload and launch the model set
if False:
    upload_and_launch(runs_upload)
#

# %% Sensitivity: partial ascertainment of both cases and deaths (Appendix 2.4.4 and 3.3)
# The baseline model refitted over a grid of two quantities fixed as data (not identifiable):
#   epsilon: case ascertainment after the intervention, p_t = (1 - epsilon) * p + epsilon
#   theta:   ascertainment of deaths among the missed cases; a fraction (1 - theta) of the deaths
#            expected among the missed cases is added to the observed deaths
# epsilon = theta = 1 reproduces the baseline.
_PARTIAL_SRC = (
    _SRC + "daily_gamma_partial_asc_semi_mechanistic_uniform_quasi-Binomial.stan"
)
eps_values = [round(0.1 * _k, 1) for _k in range(1, 11)]  # 0.1 ... 1.0
theta_values = [round(0.1 * _k, 1) for _k in range(1, 11)]  # 0.1 ... 1.0


def _tag(x):  # 0.4 -> "04", 1.0 -> "10"  (keeps basenames/dirnames free of dots)
    return f"{int(round(x * 10)):02d}"


runs_grid = [
    (
        f"Rt_daily_gamma_BVD_partial_e{_tag(_e)}_t{_tag(_th)}",
        _PARTIAL_SRC,
        {**_data_daily_gamma(7.71, 4.68), "epsilon": float(_e), "theta": float(_th)},
    )
    for _e in eps_values
    for _th in theta_values
]

# Coarse subgrid (every 0.2) first, so that the early batches span the whole surface.
_coarse = {round(0.2 * _k, 1) for _k in range(1, 6)}  # 0.2 0.4 0.6 0.8 1.0
runs_grid.sort(
    key=lambda r: 0 if (r[2]["epsilon"] in _coarse and r[2]["theta"] in _coarse) else 1
)

# Upload in batches of 10 to limit the load on the server (_BATCH = 0, 1, ...).
_BATCH, _BATCH_SIZE = 3, 10
_n_batches = -(-len(runs_grid) // _BATCH_SIZE)
_batch_runs = runs_grid[_BATCH * _BATCH_SIZE : (_BATCH + 1) * _BATCH_SIZE]

# Grid cells without a local model directory have not been launched yet.
_grid_missing = [r for r in runs_grid
                 if not os.path.isdir(os.path.join(mainstandirname, r[0]))]
if _grid_missing:
    print(colored(f"not yet launched: {len(_grid_missing)} cells -> "
                  + ", ".join(f"(eps={d['epsilon']:.1f}, theta={d['theta']:.1f})"
                              for _, _, d in _grid_missing[:8])
                  + (" ..." if len(_grid_missing) > 8 else ""), "yellow"))

print(
    colored(
        f"\nEPS/THETA GRID: {len(runs_grid)} fits "
        f"({len(eps_values)} epsilon x {len(theta_values)} theta), "
        f"batch {_BATCH + 1}/{_n_batches} -> {len(_batch_runs)} fits",
        "cyan",
    )
)
for _bn, _, _d in _batch_runs:
    print(f"  {_bn:44s}  epsilon={_d['epsilon']:.1f}  theta={_d['theta']:.1f}")

# set to True to launch this batch
if False:
    upload_and_launch(_batch_runs)
#

# %% Loading results of Stan simulations
# With _ONLY set, download the same subset that was uploaded.
all_basenames = [b for b, _, _ in (runs_upload if _ONLY else runs_all)]
if False:
    for basename in all_basenames:
        standirname = os.path.join(mainstandirname, basename)
        os.makedirs(standirname, exist_ok=True)  # curl --output-dir needs it to exist
        print(basename)

        # list remote files
        ls_out = subprocess.run(
            f"sshpass -p {remote_userPSW} ssh {remote_userID}@{remote_IP} -n -f "
            f"\"sh -c 'export LC_ALL=C LANG=C; cd Dropbox/{standirname[9:]}; ls -p ' | grep -v / \"",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        fls = ls_out.stdout.split()

        # traces, sampler logs, and the Stan code and Data/Inits JSON used remotely
        for fl in [
            fl
            for fl in fls
            if ("trace" in fl)
            or ("output" in fl)
            or fl.endswith(".stan")
            or fl.endswith(".json")
        ]:
            subprocess.run(
                [
                    "curl",
                    "-Ss",
                    "--insecure",
                    "--user",
                    f"{remote_userID}:{remote_userPSW}",
                    "--output-dir",
                    standirname,
                    "-O",
                    f"sftp://{remote_IP}/home/andrei/Dropbox/{standirname[9:]}/{fl}",
                ],
                check=False,
            )
#

# %% showing the results
tau_rep_values = [2, 4, 8]
# BASELINE = daily, gamma BVD incubation, semi-mechanistic (uniform), quasi-Binomial CFR.
baseline_basename = "Rt_daily_gamma_BVD_semi_uniform_quasiBinomial"

_IDATA_CACHE = {}


def _load_idata(basename, cache=True):
    """Load one fit as InferenceData. Cached, because the appendix tables request the same
    fits more than once and each load parses 5 chains x 12500 draws."""
    if cache and basename in _IDATA_CACHE:
        return _IDATA_CACHE[basename]
    standirname = os.path.join(mainstandirname, basename)
    paths = [str(x) for x in list(pathlib.Path(standirname).glob("trace*.csv"))]
    if not paths:
        raise FileNotFoundError(f"no trace*.csv in {standirname}")
    print(colored(basename, "red"), f"({len(paths)} chains)")
    fit = cmdstan.from_csv(paths)
    try:  # expose log_lik as the log_likelihood group for LOO
        out = az.from_cmdstanpy(posterior=fit, log_likelihood="log_lik")
    except Exception:  # weekly tau_rep ramp has no log_lik
        out = az.from_cmdstanpy(posterior=fit)
    if cache:
        _IDATA_CACHE[basename] = out
    return out


# baseline fit, used for the main figures
idata = _load_idata(baseline_basename)


# ramp-model fits for the tau_rep sensitivity figures (daily ramp fits, or the weekly ones
# if only those are present)
def _load_tau_sweep(t):
    for _bn in (f"Rt_daily_gamma_BVD_tau_{t}", f"Rt_weekly_triangular_tau_{t}"):
        try:
            return _load_idata(_bn)
        except Exception:
            continue
    raise FileNotFoundError(f"no tau_rep={t} ramp fit found (daily or weekly)")


idata_by_tau = {t: _load_tau_sweep(t) for t in tau_rep_values}

stats_ = get_stats(
    idata,
    [
        "R_pre",
        "R_post",
        "phi_rep",
        "p_check",
        "CFR_pre",
        "CFR_post",
        "OR_exposure",
        "q",
    ],
).rename(
    columns={"q2.5": "lower", "q97.5": "upper", "q25": "IQR_lower", "q75": "IQR_upper"}
)

# total missed / true cases are counts -> summarize as integers
rows_counts = []
for v in ["total_missed_cases", "total_true_cases"]:
    if v in idata.posterior:
        x = idata.posterior[v].values.reshape(-1)
        rows_counts.append(
            {
                "var": v,
                "mean": int(round(x.mean())),
                "median": int(round(np.median(x))),
                "lower": int(round(np.percentile(x, 2.5))),
                "upper": int(round(np.percentile(x, 97.5))),
            }
        )
counts_ = pd.DataFrame(rows_counts)
print(counts_.to_string(index=False))

show_stats(stats_, sig=3)
#


# %% Appendix Tables 1-4 (each = model rows x key parameter columns)
#   Table 1  daily, BVD incubation (gamma/Weibull/lognormal/triangular) x 4 structures, quasi-Binomial CFR
#   Table 2  same model set as Table 1, but the gamma-approximation CFR
#   Table 3  weekly, triangular BVD incubation, 3 structures,                     quasi-Binomial CFR
#   Table 4  daily, gamma EVD incubation, 3 structures,                           quasi-Binomial CFR
def _ci(x, d=2):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{x.mean():.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


def _loo_elpd(idata):
    """(elpd_loo, se) robust to arviz version; nan if there is no log_likelihood group."""
    try:
        L = az.loo(idata)
    except Exception:
        return float("nan"), float("nan")
    for _a in ("elpd_loo", "elpd"):
        if hasattr(L, _a):
            return float(getattr(L, _a)), float(getattr(L, "se", float("nan")))
    import re

    _m = re.search(r"elpd_loo\s+(-?\d+\.?\d*)\s+(\d+\.?\d*)", str(L))
    return (
        (float(_m.group(1)), float(_m.group(2))) if _m else (float("nan"), float("nan"))
    )


# PSIS-LOO is not reported (semi- and fully mechanistic models do not predict the same
# quantity); set to True to compute it.
_DO_LOO = False


def _model_row(basename):
    """Posterior summaries for one fit, or None when that fit is not on disk."""
    try:
        _id = _load_idata(basename)
    except Exception as _e:
        print(colored(f"  skip {basename}: {_e}", "yellow"))
        return None
    _p = _id.posterior
    _g = lambda v: _p[v].values.reshape(-1)
    _tm, _tt = _g("total_missed_cases"), _g("total_true_cases")
    _row = {  # columns of the appendix tables
        "Pre-intervention Re": _ci(_g("R_pre")),
        "Post-intervention Re": _ci(_g("R_post")),
        "Pre-intervention CFR, %": _ci(_g("CFR_pre") * 100, d=1),
        "Post-intervention CFR, %": _ci(_g("CFR_post") * 100, d=1),
        "Pre-intervention under-ascertainment, %": _ci((1 - _g("p_check")) * 100, d=1),
        "Overall under-ascertainment, %": _ci(
            _tm / _tt * 100, d=1
        ),  # missed / (observed + missed)
        "Missed cases": f"{int(round(_tm.mean()))} ({int(round(np.percentile(_tm, 2.5)))}-{int(round(np.percentile(_tm, 97.5)))})",
        "OR": _ci(_g("OR_exposure")),
    }
    if _DO_LOO:
        _e, _s = _loo_elpd(_id)
        _row["elpd_loo"] = f"{_e:.1f} ± {_s:.1f}" if not np.isnan(_e) else "-"
    return _row


def _build_table(spec, quasi_binomial, title, stem):
    """spec = [(model label, incubation label, basename)]; the _quasiBinomial tag is appended
    when quasi_binomial is True. Prints, writes <stem>.csv, and returns the DataFrame."""
    _rows = []
    for _mlab, _ilab, _bn in spec:
        _st = _model_row(f"{_bn}_quasiBinomial" if quasi_binomial else _bn)
        if _st is not None:
            _rows.append({"Model": _mlab, "Incubation period": _ilab, **_st})
    _df = pd.DataFrame(_rows)
    print(colored(f"\n=== {title} ===", "cyan"))
    print(
        _df.to_string(index=False)
        if len(_df)
        else colored("  (no fits found on disk)", "yellow")
    )
    if len(_df):
        _df.to_csv(os.path.join(mainstandirname, f"{stem}.csv"), index=False)
    return _df


_STRUCTS = [
    ("Semi-mechanistic (uniform allocation)", "semi_uniform"),
    ("Semi-mechanistic (Dirichlet prior)", "semi_dirichlet"),
    ("Fully mechanistic (VTM)", "fully_vtm"),
    ("Fully mechanistic (CV)", "fully_cv"),
]
_INCUBS = [
    ("Gamma (baseline)", "gamma_BVD"),
    ("Weibull", "Weibull_BVD"),
    ("Lognormal", "lognormal_BVD"),
    ("Triangular", "triangular"),
]

# Tables 1 and 2 share this model set and differ only in the case-fatality likelihood.
_spec_daily_bvd = [
    (_slab, _ilab, f"Rt_daily_{_i}_{_s}")
    for _slab, _s in _STRUCTS
    for _ilab, _i in _INCUBS
]
_spec_weekly_tri = [
    ("Semi-mechanistic", "Triangular", "Rt_weekly_triangular_step"),
    ("Fully mechanistic (VTM)", "Triangular", "Rt_weekly_triangular_fully_vtm"),
    ("Fully mechanistic (CV)", "Triangular", "Rt_weekly_triangular_fully_cv"),
]
_spec_gamma_evd = [
    ("Semi-mechanistic", "Gamma (EVD)", "Rt_daily_gamma_EVD_semi_uniform"),
    ("Fully mechanistic (VTM)", "Gamma (EVD)", "Rt_daily_gamma_EVD_fully_vtm"),
    ("Fully mechanistic (CV)", "Gamma (EVD)", "Rt_daily_gamma_EVD_fully_cv"),
]

appendix_table1 = _build_table(
    _spec_daily_bvd,
    True,
    "Appendix Table 1. Daily models, BVD incubation period (quasi-Binomial CFR)",
    "appendix_table1_daily_BVD_quasiBinomial",
)
appendix_table2 = _build_table(
    _spec_daily_bvd,
    False,
    "Appendix Table 2. Daily models, BVD incubation period (gamma-approximation CFR)",
    "appendix_table2_daily_BVD_gammaApprox",
)
appendix_table3 = _build_table(
    _spec_weekly_tri,
    True,
    "Appendix Table 3. Weekly scale, triangular BVD incubation period (quasi-Binomial CFR)",
    "appendix_table3_weekly_triangular",
)
appendix_table4 = _build_table(
    _spec_gamma_evd,
    True,
    "Appendix Table 4. Daily models, gamma EVD incubation period (quasi-Binomial CFR)",
    "appendix_table4_gamma_EVD",
)

_APPENDIX = [
    ("Table1_dailyBVD_qBinom", "1", appendix_table1),
    ("Table2_dailyBVD_gammaApprox", "2", appendix_table2),
    ("Table3_weekly_triangular", "3", appendix_table3),
    ("Table4_gamma_EVD", "4", appendix_table4),
]

# one workbook, one sheet per appendix table
with pd.ExcelWriter(os.path.join(mainstandirname, "appendix_tables.xlsx")) as _xl:
    for _sheet, _num, _df in _APPENDIX:
        if len(_df):
            _df.to_excel(_xl, sheet_name=_sheet[:31], index=False)

# flat concatenation of all four, for grepping/plotting across the whole model set
comparison_table = pd.concat(
    [_df.assign(Table=_num) for _sheet, _num, _df in _APPENDIX if len(_df)],
    ignore_index=True,
)
comparison_table = comparison_table[
    ["Table"] + [c for c in comparison_table.columns if c != "Table"]
]
comparison_table.to_csv(
    os.path.join(mainstandirname, "model_comparison_table.csv"), index=False
)
print(
    colored(
        f"\nwrote appendix_tables.xlsx (4 sheets) + 4 CSVs + model_comparison_table.csv "
        f"[{len(comparison_table)} rows]",
        "green",
    )
)

# PSIS-LOO across every loaded fit that exposes log_lik (only when _DO_LOO)
if _DO_LOO:
    _loo_idatas = {
        _k: _v for _k, _v in _IDATA_CACHE.items() if hasattr(_v, "log_likelihood")
    }
    try:
        loo_compare = az.compare(_loo_idatas)
        print("\n=== PSIS-LOO comparison ===")
        print(loo_compare.to_string())
        loo_compare.to_csv(os.path.join(mainstandirname, "model_loo_compare.csv"))
    except Exception as _e:
        print(colored(f"az.compare failed: {_e}", "yellow"))
#
# %% Plotting the results
from matplotlib.ticker import MultipleLocator

# --- forward-looking reproduction number over weeks ------------------------
df_Rt = get_stats(idata, ["Rt_infection"]).rename(
    columns={"q2.5": "lower", "q97.5": "upper", "q25": "IQR_lower", "q75": "IQR_upper"}
)
df_Rt["week"] = df_Rt["time"].astype(int) + min_week - 1  # week_index -> calendar week
df_Rt = df_Rt.sort_values("week")
df_Rt = df_Rt[
    df_Rt["week"] <= int(df["week"].max())
]  # drop the 3 tail-zero weeks for plotting

fig, ax = plt.subplots(1, 1, figsize=[3.4, 2.2])  # single-column width

# R(t): posterior mean + 95% and IQR credible bands (steps-mid: jump on dashed line)
ax.plot(df_Rt["week"], df_Rt["mean"], c="k", lw=1.2, zorder=4, drawstyle="steps-mid")
ax.fill_between(
    df_Rt["week"],
    df_Rt["lower"],
    df_Rt["upper"],
    color="C7",
    alpha=0.15,
    zorder=2,
    step="mid",
)
ax.fill_between(
    df_Rt["week"],
    df_Rt["IQR_lower"],
    df_Rt["IQR_upper"],
    color="C7",
    alpha=0.20,
    zorder=2,
    step="mid",
)

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(bottom=0)
ymax = ax.get_ylim()[1]

# total weekly cases as white bars, scaled to the R axis
epi_max = int(df["count_total"].max())
case_unit = 0.45 * ymax / epi_max
ax.bar(
    df["week"],
    df["count_total"] * case_unit,
    width=0.9,
    align="center",
    facecolor="0.7",
    edgecolor="k",
    linewidth=0.5,
    zorder=0,
)

# --- scale bar (top-right): height = `scale_n` cases ------------------------
xmn, xmx = ax.get_xlim()
scale_n = 10
x_sb = xmn + 0.96 * (xmx - xmn)
y_top = 0.97 * ymax
ax.plot(
    [x_sb, x_sb],
    [y_top - scale_n * case_unit, y_top],
    color="k",
    lw=2.5,
    solid_capstyle="butt",
    zorder=6,
)
ax.text(
    x_sb - 0.02 * (xmx - xmn),
    y_top,
    f"{scale_n} cases",
    fontsize=6,
    ha="right",
    va="top",
)

# --- intervention changepoint (dashed marker) ------------------------------
ax.vlines(
    intervention_effect_week - 0.5, 0, ymax, color="k", ls="dashed", lw=0.6, zorder=3
)

ax.set_xlim(xmn, xmx)
ax.set_ylim(0, ymax)
ax.set_ylabel("Effective reproduction number", fontsize=8, labelpad=4)
ax.set_xlabel("Epidemiological week in 2007", fontsize=8, labelpad=3)
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.tick_params(labelsize=7)

plt.savefig(f"{figuresdir}/R_t_Bundibugyo.pdf", format="pdf", bbox_inches="tight")
plt.show()
#
# %% Appendix Figure 1: R(t), CFR(t) and under-ascertainment
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# --- panel A data: forward-looking reproduction number ---------------------
df_Rt = get_stats(idata, ["Rt_infection"]).rename(
    columns={"q2.5": "lower", "q97.5": "upper", "q25": "IQR_lower", "q75": "IQR_upper"}
)
df_Rt["week"] = df_Rt["time"].astype(int) + min_week - 1
df_Rt = df_Rt.sort_values("week")
df_Rt = df_Rt[
    df_Rt["week"] <= int(df["week"].max())
]  # drop the 3 tail-zero weeks for plotting

# --- panel B data: weekly CFR reconstructed from the q_week schedule --------
# CFR_t = q_week * CFR_pre : CFR_pre before t_int, (CFR_pre+CFR_post)/2 at the
# transition week (t_int-1), CFR_post from t_int onward  -> continuous ramp.
post = idata.posterior
cfr_pre = post["CFR_pre"].values.reshape(-1)
cfr_post = post["CFR_post"].values.reshape(-1)
wk_idx = df["week_index"].values[:, None]  # (W, 1)
cfr_t = np.where(
    wk_idx >= changepoint,
    cfr_post[None, :],
    np.where(
        wk_idx == changepoint - 1, 0.5 * (cfr_pre + cfr_post)[None, :], cfr_pre[None, :]
    ),
)  # (W, ndraws)
cfr_mean = cfr_t.mean(1)
cfr_lo, cfr_hi = np.percentile(cfr_t, [2.5, 97.5], axis=1)
cfr_qlo, cfr_qhi = np.percentile(cfr_t, [25, 75], axis=1)

# observed weekly CFR: Jeffreys Beta(0.5,0.5) prior -> Beta(0.5+deaths, 0.5+survivors)
from scipy.stats import beta as _beta

obs = df.loc[df["count_total"] > 0].copy()
a_obs = 0.5 + obs["count_fatal"].values
b_obs = 0.5 + obs["count_non_fatal"].values
obs_mean = a_obs / (a_obs + b_obs)  # posterior mean
obs_lo = _beta.ppf(0.025, a_obs, b_obs)  # 95% CrI
obs_hi = _beta.ppf(0.975, a_obs, b_obs)

# --- panel C data: under-ascertainment rate = (1 - p_asc) * 100 ------------
p_asc_draws = post["p_asc"].values.reshape(-1, post["p_asc"].shape[-1])[
    :, : df.shape[0]
]  # (ndraws, W); model carries W+3
ua = (1 - p_asc_draws) * 100
ua_mean = ua.mean(0)
ua_lo, ua_hi = np.percentile(ua, [2.5, 97.5], axis=0)
ua_qlo, ua_qhi = np.percentile(ua, [25, 75], axis=0)

fig, ((axA, axC), (axB, axD)) = plt.subplots(2, 2, figsize=[6.8, 4.2])

# ===== Panel A: R(t) ========================================================
# steps-mid: each week's posterior is held over [week-0.5, week+0.5], so the
# R_pre -> R_post jump lands on the dashed line at intervention_effect_week-0.5
axA.plot(df_Rt["week"], df_Rt["mean"], c="k", lw=1.2, zorder=4, drawstyle="steps-mid")
axA.fill_between(
    df_Rt["week"],
    df_Rt["lower"],
    df_Rt["upper"],
    color="C7",
    alpha=0.15,
    zorder=2,
    step="mid",
)
axA.fill_between(
    df_Rt["week"],
    df_Rt["IQR_lower"],
    df_Rt["IQR_upper"],
    color="C7",
    alpha=0.20,
    zorder=2,
    step="mid",
)

axA.spines[["top", "right"]].set_visible(False)
axA.set_ylim(bottom=0)
ymax = axA.get_ylim()[1]

epi_max = int(df["count_total"].max())
case_unit = 0.45 * ymax / epi_max
axA.bar(
    df["week"],
    df["count_fatal"] * case_unit,
    width=0.9,
    align="center",
    facecolor="k",
    edgecolor="k",
    linewidth=0.5,
    zorder=0,
    label="Fatal",
)
axA.bar(
    df["week"],
    df["count_non_fatal"] * case_unit,
    width=0.9,
    align="center",
    bottom=df["count_fatal"] * case_unit,
    facecolor="0.7",
    edgecolor="k",
    linewidth=0.5,
    zorder=0,
    label="Non-fatal",
)

xmn, xmx = axA.get_xlim()
scale_n = 10
x_sb = xmn + 0.96 * (xmx - xmn)
y_top = 0.97 * ymax
axA.plot(
    [x_sb, x_sb],
    [y_top - scale_n * case_unit, y_top],
    color="k",
    lw=2.5,
    solid_capstyle="butt",
    zorder=6,
)
axA.text(
    x_sb - 0.02 * (xmx - xmn),
    y_top,
    f"{scale_n} cases",
    fontsize=6,
    ha="right",
    va="top",
)

axA.vlines(
    intervention_effect_week - 0.5, 0, ymax, color="k", ls="dashed", lw=0.6, zorder=3
)
axA.set_ylim(0, ymax)
axA.set_ylabel(r"$\mathrm{R_\text{eff}}$", fontsize=9, labelpad=4)
axA.tick_params(labelsize=7, labelbottom=False)

# legend above panel A: two-column horizontal, outside the axes
axA.legend(
    handles=[
        Patch(facecolor="k", edgecolor="k", label="Fatal"),
        Patch(facecolor="0.7", edgecolor="k", label="Non-fatal"),
    ],
    ncol=2,
    frameon=False,
    fontsize=7,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    handlelength=1.2,
    columnspacing=1.5,
    handletextpad=0.5,
    borderaxespad=0,
)

# ===== Panel B: CFR(t) ======================================================
# observed weekly CFR (Jeffreys Beta posterior): point = mean, whiskers = 95% CrI
axB.errorbar(
    obs["week"],
    obs_mean * 100,
    yerr=[(obs_mean - obs_lo) * 100, (obs_hi - obs_mean) * 100],
    fmt="o",
    ms=3,
    mfc="0.5",
    mec="0.3",
    mew=0.4,
    ecolor="0.5",
    elinewidth=0.6,
    capsize=0,
    zorder=1,
)

# model estimate (3-level step) + 95% and IQR bands, in %
# steps-mid: intermediate level occupies the single week before the dashed line
axB.plot(df["week"], cfr_mean * 100, c="k", lw=1.2, zorder=4, drawstyle="steps-mid")
axB.fill_between(
    df["week"],
    cfr_lo * 100,
    cfr_hi * 100,
    color="C7",
    alpha=0.15,
    zorder=2,
    step="mid",
)
axB.fill_between(
    df["week"],
    cfr_qlo * 100,
    cfr_qhi * 100,
    color="C7",
    alpha=0.20,
    zorder=2,
    step="mid",
)

axB.spines[["top", "right"]].set_visible(False)
axB.vlines(
    intervention_effect_week - 0.5, 0, 100, color="k", ls="dashed", lw=0.6, zorder=3
)

axB.set_ylim(0, 100)
axB.set_yticks([0, 25, 50, 75, 100])
axB.set_ylabel("CFR, %", fontsize=8, labelpad=4)
axB.set_xlabel("Epidemiological week in 2007", fontsize=8, labelpad=3)
axB.set_xlim(xmn, xmx)
axB.xaxis.set_major_locator(MultipleLocator(4))
axB.tick_params(labelsize=7)

# ===== Panel C: under-ascertainment (t) =====================================
# steps-mid: p_asc=1 (ua=0) from week_index >= changepoint, so ua drops to 0
# exactly at the dashed line (intervention_effect_week-0.5)
axC.plot(df["week"], ua_mean, c="k", lw=1.2, zorder=4, drawstyle="steps-mid")
axC.fill_between(df["week"], ua_lo, ua_hi, color="C7", alpha=0.15, zorder=2, step="mid")
axC.fill_between(
    df["week"], ua_qlo, ua_qhi, color="C7", alpha=0.20, zorder=2, step="mid"
)

# The curve is the time-varying rate (about 19% before the intervention, 0 after); the dashed
# line is the overall rate, missed / (observed + missed), quoted in the main text.
_ua_overall = (
    post["total_missed_cases"].values.reshape(-1)
    / post["total_true_cases"].values.reshape(-1)
    * 100
).mean()
axC.axhline(_ua_overall, ls=(0, (4, 2)), lw=0.8, color="k", zorder=5)
axC.annotate(
    f"overall {_ua_overall:.1f}%",
    xy=(xmx, _ua_overall), xytext=(-2, 2), textcoords="offset points",
    ha="right", va="bottom", fontsize=6,
)

axC.spines[["top", "right"]].set_visible(False)
axC.set_ylim(bottom=0)
axC.vlines(
    intervention_effect_week - 0.5,
    0,
    axC.get_ylim()[1],
    color="k",
    ls="dashed",
    lw=0.6,
    zorder=3,
)
axC.set_ylabel("Under-ascertainment, %", fontsize=8, labelpad=4)
axC.set_xlabel("Epidemiological week in 2007", fontsize=8, labelpad=3)
axC.set_xlim(xmn, xmx)
axC.xaxis.set_major_locator(MultipleLocator(4))
axC.tick_params(labelsize=7)

# ===== Panel D: empty =====
axD.axis("off")

# panel letters
for a, lab in [(axA, "A"), (axB, "B"), (axC, "C")]:
    a.text(
        -0.16,
        1.02,
        lab,
        transform=a.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

plt.savefig(
    f"{figuresdir}/R_CFR_ascertainment_Bundibugyo.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
#
# %% Forest plot of posterior estimates (not used in the paper)
post = idata.posterior


def _summ(x):
    x = np.asarray(x).reshape(-1)
    return x.mean(), np.percentile(x, 2.5), np.percentile(x, 97.5)


# --- rows (label, mean, lo, hi), top to bottom -----------------------------
rows_R = [
    (r"$\mathrm{R_\text{eff}}$ (preinterv.)", *_summ(post["R_pre"].values)),
    (r"$\mathrm{R_\text{eff}}$ (postinterv.)", *_summ(post["R_post"].values)),
]
rows_pct = [
    ("CFR (preinterv.), %", *_summ(post["CFR_pre"].values * 100)),
    ("CFR (postinterv.), %", *_summ(post["CFR_post"].values * 100)),
    ("Under-ascertainment, %", *_summ((1 - post["p_check"].values) * 100)),
]

pct_xmax = 50  # x-axis limit for percentages


def forest_panel(ax, rows, xlim, ref=None, xticks=None, xticklabels=None, fmt="{:.1f}"):
    ys = np.arange(len(rows))[::-1]  # top row = highest y
    if ref is not None:
        ax.axvline(ref, color="k", ls="dashed", lw=0.8, zorder=1)
    for y, (lab, m, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="k", lw=1.3, solid_capstyle="butt", zorder=2)
        ax.plot(m, y, "o", mfc="white", mec="k", mew=1.3, ms=5, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # right-hand "Mean (95% CrI)" text column: centered, fixed axes-fraction x
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    xc = 1.28
    for y, (lab, m, lo, hi) in zip(ys, rows):
        ax.text(
            xc,
            y,
            f"{fmt.format(m)} ({fmt.format(lo)}–{fmt.format(hi)})",
            transform=trans,
            fontsize=7,
            va="center",
            ha="center",
            clip_on=False,
        )
    return xc, trans


fig, (axT, axB) = plt.subplots(
    2, 1, figsize=[5.2, 1.9], gridspec_kw={"height_ratios": [2, 3], "hspace": 0.1}
)

# top: reproduction numbers (own scale, reference at R=1)
xr_T, trans_T = forest_panel(
    axT, rows_R, xlim=(0, 3), ref=1.0, xticks=[0, 1, 2, 3], fmt="{:.2f}"
)
axT.text(
    xr_T,
    len(rows_R) - 0.1,
    "Mean (95% CrI)",
    transform=trans_T,
    fontsize=7,
    fontweight="bold",
    va="bottom",
    ha="center",
    clip_on=False,
)

# bottom: percentages (CFR, under-ascertainment)
xticks_B = np.arange(0, pct_xmax + 1, 10)
xr_B, trans_B = forest_panel(
    axB, rows_pct, xlim=(0, pct_xmax), xticks=xticks_B, fmt="{:.1f}"
)
axB.text(
    xr_B,
    len(rows_pct) - 0.1,
    "Mean (95% CrI)",
    transform=trans_B,
    fontsize=7,
    fontweight="bold",
    va="bottom",
    ha="center",
    clip_on=False,
)

plt.savefig(f"{figuresdir}/forest_Bundibugyo.pdf", format="pdf", bbox_inches="tight")
plt.show()
#
# %% Figure 1: boxplots of posterior estimates, pre- (grey) and post-intervention (white)
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

post = idata.posterior

PRE_COLOR, POST_COLOR = "0.8", "white"  # pre-intervention grey, post white


def _summ5(x):
    x = np.asarray(x).reshape(-1)
    return (
        x.mean(),  # mean   (filled dot)
        np.percentile(x, 50),  # median (vertical line in box)
        np.percentile(x, 2.5),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.percentile(x, 97.5),
    )


def box_panel(
    ax,
    rows,
    xlim,
    yticks,
    yticklabels,
    ref=None,
    xticks=None,
    xticklabels=None,
    fmt="{:.1f}",
    bh=0.5,
):
    # rows: list of (summ, facecolor), top to bottom, evenly spaced
    ys = np.arange(len(rows))[::-1]
    if ref is not None:
        ax.axvline(ref, color="k", ls="dotted", lw=0.8, zorder=1)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    xc = 1.28
    for y, (s, fc) in zip(ys, rows):
        m, med, lo, q1, q3, hi = s
        # 95% CrI whiskers (no caps)
        ax.plot([lo, q1], [y, y], color="k", lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([q3, hi], [y, y], color="k", lw=1.0, solid_capstyle="butt", zorder=2)
        # IQR box
        ax.add_patch(
            Rectangle(
                (q1, y - bh / 2),
                q3 - q1,
                bh,
                facecolor=fc,
                edgecolor="k",
                lw=1.0,
                zorder=3,
            )
        )
        # median vertical line + mean dot
        ax.plot([med, med], [y - bh / 2, y + bh / 2], color="k", lw=1.0, zorder=4)
        ax.plot(m, y, "o", mfc="k", mec="k", ms=4, zorder=5)
        # right-hand Mean (95% CrI) text (one per row, evenly spaced)
        ax.text(
            xc,
            y,
            f"{fmt.format(m)} ({fmt.format(lo)}–{fmt.format(hi)})",
            transform=trans,
            fontsize=7,
            va="center",
            ha="center",
            clip_on=False,
        )
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return xc, trans


pct_xmax = 50  # x-axis limit for percentages

# top: reproduction number, pre (grey) over post (white), one label centred between them
rows_R = [
    (_summ5(post["R_pre"].values), PRE_COLOR),
    (_summ5(post["R_post"].values), POST_COLOR),
]
# bottom: CFR pre/post + under-ascertainment (pre only, no post)
rows_pct = [
    (_summ5(post["CFR_pre"].values * 100), PRE_COLOR),
    (_summ5(post["CFR_post"].values * 100), POST_COLOR),
    (_summ5((1 - post["p_check"].values) * 100), PRE_COLOR),
]

fig, (axT, axB) = plt.subplots(
    2, 1, figsize=[5.2, 1.9], gridspec_kw={"height_ratios": [2, 3], "hspace": 0.1}
)

# top: reproduction numbers (own scale, dotted reference at R=1)
xr_T, trans_T = box_panel(
    axT,
    rows_R,
    xlim=(0, 3),
    yticks=[0.5],
    yticklabels=["Reproduction\nnumber"],
    ref=1.0,
    xticks=[0, 1, 2, 3],
    fmt="{:.2f}",
)
axT.text(
    xr_T,
    axT.get_ylim()[1] + 0.4,
    "Mean (95% CrI)",
    transform=trans_T,
    fontsize=7,
    fontweight="bold",
    va="bottom",
    ha="center",
    clip_on=False,
)

# legend on top: grey = pre-intervention, white = post-intervention (horizontal)
axT.legend(
    handles=[
        Patch(facecolor=PRE_COLOR, edgecolor="k", label="Preintervention"),
        Patch(facecolor=POST_COLOR, edgecolor="k", label="Postintervention"),
    ],
    ncol=2,
    frameon=False,
    fontsize=7,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.15),
    handlelength=1.3,
    columnspacing=1.5,
    handletextpad=0.5,
    borderaxespad=0,
)

# bottom: CFR pre/post with one label, then under-ascertainment
xticks_B = np.arange(0, pct_xmax + 1, 10)
xr_B, trans_B = box_panel(
    axB,
    rows_pct,
    xlim=(0, pct_xmax),
    yticks=[1.5, 0],
    yticklabels=["CFR, %", "Under-\nascertainment, %"],
    xticks=xticks_B,
    fmt="{:.1f}",
)
plt.savefig(
    f"{figuresdir}/forest_box_Bundibugyo.pdf", format="pdf", bbox_inches="tight"
)
plt.show()
#

# %% Prior vs posterior comparison
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy import special
from scipy.stats import gaussian_kde

rng = np.random.default_rng(42)
N = 300_000

# --- prior samples (respecting ordered constraint via rejection) ---
# log_R[2]=R_pre ~ normal(log(1.5), 1.0), log_R[1]=R_post ~ normal(log(1.5), 1.0), ordered
lr_post = rng.normal(np.log(1.5), 1.0, N)
lr_pre = rng.normal(np.log(1.5), 1.0, N)
ok_R = lr_post < lr_pre
prior_R_pre = np.exp(lr_pre[ok_R])
prior_R_post = np.exp(lr_post[ok_R])

# logit_CFR ordered: logit_CFR[1]=post < logit_CFR[2]=pre
lc1 = rng.normal(0, 1.5, N)
lc2 = rng.normal(0, 1.5, N)
ok_C = lc1 < lc2
prior_CFR_pre = special.expit(lc2[ok_C]) * 100
prior_CFR_post = special.expit(lc1[ok_C]) * 100
# p_check ~ Beta(6,4) → under-ascertainment = (1-p_check)*100
prior_unasc = (1 - rng.beta(6, 4, N)) * 100

# inv_sqrt_phi_rep ~ exp(1) → phi_rep = 1/x^2
prior_phi = 1 / rng.exponential(1, N) ** 2

# --- posterior samples ---
post = idata.posterior


def _flat(var):
    return post[var].values.reshape(-1)


panels = [
    # row 1: pre-intervention across columns
    (
        r"A. Preintervention $\mathrm{R}_\mathrm{eff}$",
        _flat("R_pre"),
        prior_R_pre,
        (0, 4),
        "",
    ),
    (f"C. Preintervention CFR, %", _flat("CFR_pre") * 100, prior_CFR_pre, (0, 80), ""),
    (
        "E. Under-ascertainment, %",
        (1 - _flat("p_check")) * 100,
        prior_unasc,
        (0, 70),
        "",
    ),
    # row 2: post-intervention across columns
    (
        r"B. Postintervention $\mathrm{R}_\mathrm{eff}$",
        _flat("R_post"),
        prior_R_post,
        (0, 4),
        "",
    ),
    (
        f"D. Postintervention CFR, %",
        _flat("CFR_post") * 100,
        prior_CFR_post,
        (0, 80),
        "",
    ),
    ("F. $\\phi$ (overdispersion)", _flat("phi_rep"), prior_phi, (0, 15), ""),
]

fig, axes = plt.subplots(
    2, 3, figsize=[6.8, 2.6], gridspec_kw={"hspace": 0.15, "wspace": 0.15}
)
axes = axes.flatten()

for ax, (title, post_s, prior_s, xlim, xlabel) in zip(axes, panels):
    x = np.linspace(xlim[0], xlim[1], 300)

    # posterior KDE
    ps = post_s[np.isfinite(post_s)]
    kde_post = gaussian_kde(ps, bw_method="scott")
    y_post = kde_post(x)
    ax.fill_between(x, y_post, color="0.75")
    ax.plot(x, y_post, color="0.5", lw=0.6)

    # prior KDE
    prs = prior_s[np.isfinite(prior_s) & (prior_s < xlim[1] * 5)]
    kde_prior = gaussian_kde(prs, bw_method="scott")
    ax.plot(x, kde_prior(x), "k--", lw=1.2)

    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, fontsize=8, loc="left", pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=7)

legend_handles = [
    mpatches.Patch(facecolor="0.75", edgecolor="0.5", lw=0.6, label="Posterior"),
    mlines.Line2D([], [], color="k", ls="--", lw=1.2, label="Prior"),
]
fig.legend(
    handles=legend_handles,
    ncol=2,
    frameon=False,
    fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    handlelength=1.0,
    columnspacing=0.8,
    handletextpad=0.4,
)

plt.savefig(
    f"{figuresdir}/prior_posterior_Bundibugyo.pdf", format="pdf", bbox_inches="tight"
)
plt.show()
#

# %% Appendix Figure 2: tau_rep sweep (points and whiskers, coloured by tau_rep)
# Each parameter for the baseline (step) model and the three ramp-model fits.
# Point = mean, thin line = 95% CrI, thick line = IQR.
from matplotlib.lines import Line2D

tau_colors = {2: "C0", 4: "C1", 8: "C2"}  # one colour per tau_rep


def whisker_panel(ax, getters, xlim, group_labels, series, ref=None, xticks=None):
    # series: list of (idata, colour); rows within a group from top to bottom
    n = len(getters)
    centers = np.arange(n)[::-1]  # first group at the top
    k = len(series)
    offs = np.linspace(0.30, -0.30, k) if k > 1 else [0.0]
    if ref is not None:
        ax.axvline(ref, color="k", ls="dotted", lw=0.8, zorder=1)
    for c, getter in zip(centers, getters):
        for (idata_s, col), off in zip(series, offs):
            x = np.asarray(getter(idata_s.posterior)).reshape(-1)
            m = x.mean()
            lo, q1, q3, hi = np.percentile(x, [2.5, 25, 75, 97.5])
            y = c + off
            ax.plot(
                [lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2
            )
            ax.plot(
                [q1, q3], [y, y], color=col, lw=2.6, solid_capstyle="butt", zorder=3
            )
            ax.plot(m, y, "o", mfc="white", mec=col, mew=1.0, ms=4, zorder=4)
    ax.set_yticks(centers)
    ax.set_yticklabels(group_labels, fontsize=8)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


getters_R = [lambda p: p["R_pre"].values, lambda p: p["R_post"].values]
labels_R = [
    r"Preintervention $\mathrm{R}_\mathrm{eff}$",
    r"Postintervention $\mathrm{R}_\mathrm{eff}$",
]

getters_pct = [
    lambda p: p["CFR_pre"].values * 100,
    lambda p: p["CFR_post"].values * 100,
    lambda p: (1 - p["p_check"].values) * 100,
]
labels_pct = [
    "Preintervention CFR, %",
    "Postintervention CFR, %",
    "Under-ascertainment, %",
]

# series: baseline (step model, black) + the three ramp-model tau_rep fits
series = [(idata, "k")] + [(idata_by_tau[t], tau_colors[t]) for t in tau_rep_values]
series_labels = ["Baseline (step)"] + [
    rf"$\tau_\mathrm{{rep}}={t}$" for t in tau_rep_values
]


def _series_legend(ax):
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=col,
                lw=2.4,
                marker="o",
                mfc="white",
                mec=col,
                mew=1.0,
                ms=4,
                label=lab,
            )
            for (_, col), lab in zip(series, series_labels)
        ],
        ncol=4,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0,
    )


pct_xmax = 50
fig, (axT, axB) = plt.subplots(
    2, 1, figsize=[5.0, 3.6], gridspec_kw={"height_ratios": [2, 3], "hspace": 0.25}
)
whisker_panel(
    axT,
    getters_R,
    xlim=(0, 3),
    group_labels=labels_R,
    series=series,
    ref=1.0,
    xticks=[0, 1, 2, 3],
)
whisker_panel(
    axB,
    getters_pct,
    xlim=(0, pct_xmax),
    group_labels=labels_pct,
    series=series,
    xticks=np.arange(0, pct_xmax + 1, 10),
)
_series_legend(axT)

plt.savefig(
    f"{figuresdir}/sensitivity_tau_rep_Bundibugyo.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
#

# %% Appendix Figure 3: tau_rep sweep for phi, OR and total missed cases
# Same style as Appendix Figure 2, one parameter per panel since the x-scales differ.
getters_phi = [lambda p: p["phi_rep"].values]
getters_or = [lambda p: p["OR_exposure"].values]
getters_miss = [lambda p: p["total_missed_cases"].values]


def _sweep_xmax(getter, pad=1.08, hard=0.0):
    hi = max(
        np.percentile(np.asarray(getter(s.posterior)).reshape(-1), 97.5)
        for s, _ in series
    )
    return max(hi, hard) * pad


fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=[5.0, 3.3], gridspec_kw={"hspace": 0.4}
)

whisker_panel(
    ax1,
    getters_phi,
    xlim=(0, _sweep_xmax(getters_phi[0])),
    group_labels=[r"Overdispersion, $\phi$"],
    series=series,
)
whisker_panel(
    ax2,
    getters_or,
    xlim=(0, _sweep_xmax(getters_or[0], hard=1.0)),
    group_labels=["Odds ratio (post/pre)"],
    series=series,
    ref=1.0,
)
whisker_panel(
    ax3,
    getters_miss,
    xlim=(0, _sweep_xmax(getters_miss[0], hard=13)),
    group_labels=["Total missed cases"],
    series=series,
)


_series_legend(ax1)

plt.savefig(
    f"{figuresdir}/phi_OR_missed_Bundibugyo.pdf", format="pdf", bbox_inches="tight"
)
plt.show()
#

# %% Appendix Figure 6: model fit (A) and posterior predictive coverage (B)
# A: expected weekly cases p_x*E[c_x] (posterior mean, 50% and 95% CrI; parameter uncertainty only)
# B: 95% posterior predictive interval (including observation noise); week 1 (index case) excluded
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

post = idata.posterior
# The three trailing zero weeks enter the likelihood and are plotted too (predicted decline
# after the last observed case).
W_ = df.shape[0] + 3
weeks = np.concatenate([df["week"].values, df["week"].values[-1] + np.arange(1, 4)])
obs = np.concatenate([df["count_total"].values, np.zeros(3, dtype=int)])

# --- panel A: expected reported cases (no observation noise) ---
cpm = post["case_pred_mean"].values.reshape(-1, W_)  # p_x*E[c_x] (draws, W+3)
emean = cpm.mean(0)
elo, ehi = np.percentile(cpm, [2.5, 97.5], axis=0)
eq25, eq75 = np.percentile(cpm, [25, 75], axis=0)
keep = np.arange(W_) >= 1  # drop index week
emean_p, elo_p, ehi_p, eq25_p, eq75_p = (
    np.where(keep, a, np.nan) for a in (emean, elo, ehi, eq25, eq75)
)

# --- panel B: posterior predictive interval + coverage ---
cp = post["case_pred"].values.reshape(-1, W_)  # posterior predictive (draws, W+3)
pred_lo, pred_hi = np.percentile(cp, [2.5, 97.5], axis=0)
mask = np.arange(W_) >= 1  # skip week 1 (index case)
inside = (obs >= pred_lo) & (obs <= pred_hi)
n_in, n_tot = int(inside[mask].sum()), int(mask.sum())
print(f"95% predictive coverage: {n_in}/{n_tot} weeks ({100 * n_in / n_tot:.0f}%)")
print("weeks outside the 95% interval:", list(weeks[mask][~inside[mask]]))

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=[3.4, 4.2], sharex=True, gridspec_kw={"hspace": 0.15}
)

# ===== Panel A: expected vs observed =====
axA.bar(
    weeks,
    obs,
    width=0.9,
    align="center",
    facecolor="0.8",
    edgecolor="k",
    linewidth=0.4,
    zorder=1,
)
axA.fill_between(weeks, elo_p, ehi_p, color="C7", alpha=0.25, zorder=2)
axA.fill_between(weeks, eq25_p, eq75_p, color="C7", alpha=0.40, zorder=2)
axA.plot(weeks, emean_p, color="k", lw=1.3, marker="o", ms=2.5, zorder=4)
axA.spines[["top", "right"]].set_visible(False)
axA.set_ylim(bottom=0)
ymaxA = axA.get_ylim()[1]
axA.vlines(
    intervention_effect_week - 0.5, 0, ymaxA, color="k", ls="dashed", lw=0.6, zorder=3
)
axA.set_ylim(0, ymaxA)
axA.set_ylabel("Weekly cases", fontsize=8, labelpad=4)
axA.tick_params(labelsize=7)
axA.legend(
    handles=[
        Patch(facecolor="0.8", edgecolor="k", lw=0.4, label="Observed"),
        Line2D([], [], color="k", lw=1.3, marker="o", ms=2.5, label="Expected"),
    ],
    frameon=False,
    fontsize=7,
    loc="upper left",
)

# ===== Panel B: 95% predictive interval (coverage) =====
axB.bar(
    weeks,
    obs,
    width=0.9,
    align="center",
    facecolor="0.8",
    edgecolor="k",
    linewidth=0.4,
    zorder=1,
)
axB.vlines(weeks[mask], pred_lo[mask], pred_hi[mask], color="k", lw=0.9, zorder=3)
axB.spines[["top", "right"]].set_visible(False)
axB.set_ylim(bottom=0)
ymaxB = axB.get_ylim()[1]
axB.vlines(
    intervention_effect_week - 0.5, 0, ymaxB, color="k", ls="dashed", lw=0.6, zorder=2
)
axB.set_ylim(0, ymaxB)
axB.set_ylabel("Weekly cases", fontsize=8, labelpad=4)
axB.set_xlabel("Epidemiological week in 2007", fontsize=8, labelpad=3)
# explicit ticks: the series ends at week 55
axB.set_xticks(np.arange(32, weeks.max() + 1, 4))
axB.set_xlim(weeks.min() - 0.8, weeks.max() + 0.8)
axB.tick_params(labelsize=7)
axB.legend(
    handles=[
        Patch(facecolor="0.8", edgecolor="k", lw=0.4, label="Observed"),
        Line2D([], [], color="k", lw=0.9, label="95% predictive interval"),
    ],
    frameon=False,
    fontsize=7,
    loc="upper left",
)

# panel letters (same style as the R(t)/CFR figure)
for a, lab in [(axA, "A"), (axB, "B")]:
    a.text(
        -0.16,
        1.02,
        lab,
        transform=a.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

plt.savefig(
    f"{figuresdir}/fit_ppc_cases_Bundibugyo.pdf", format="pdf", bbox_inches="tight"
)
plt.show()


# %% Sensitivity analysis for partial ascertainment: download the grid fits
def download_fits(basenames):
    for basename in basenames:
        standirname = os.path.join(mainstandirname, basename)
        os.makedirs(standirname, exist_ok=True)  # curl --output-dir needs it to exist
        ls_out = subprocess.run(
            f"sshpass -p {remote_userPSW} ssh {remote_userID}@{remote_IP} -n -f "
            f"\"sh -c 'export LC_ALL=C LANG=C; cd Dropbox/{standirname[9:]}; ls -p ' | grep -v / \"",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        want = [
            f
            for f in ls_out.stdout.split()
            if ("trace" in f)
            or ("output" in f)
            or f.endswith(".stan")
            or f.endswith(".json")
        ]
        n_tr = sum("trace" in f for f in want)
        print(
            f"{basename:46s} {len(want):3d} files ({n_tr} traces)"
            + ("" if n_tr else colored("   <- no traces yet", "yellow"))
        )
        for fl in want:
            subprocess.run(
                [
                    "curl",
                    "-Ss",
                    "--insecure",
                    "--user",
                    f"{remote_userID}:{remote_userPSW}",
                    "--output-dir",
                    standirname,
                    "-O",
                    f"sftp://{remote_IP}/home/andrei/Dropbox/{standirname[9:]}/{fl}",
                ],
                check=False,
            )


# Download every grid cell that has been launched (a local model directory exists).
_grid_done = [
    r for r in runs_grid if os.path.isdir(os.path.join(mainstandirname, r[0]))
]
print(
    colored(
        f"downloading {len(_grid_done)} launched grid fits "
        f"(of {len(runs_grid)} in the grid)",
        "cyan",
    )
)
if False:
    download_fits([b for b, _, _ in _grid_done])
#
# %% Appendix Figure 4: heatmaps over the (epsilon, theta) grid
# Rows: pre-/post-intervention Re, pre-/post-intervention CFR, overall under-ascertainment of
# cases and of deaths. Cells show the posterior mean; hatched cells are not fitted.
from matplotlib.colors import Normalize

_obs_deaths = int(df["count_fatal"].sum())  # 37 ascertained deaths


def _grid_stats(basename):
    """Posterior means of the six plotted quantities; None when the fit is not on disk.
    cache=False: the grid holds many fits and we only need six scalars from each."""
    try:
        _p = _load_idata(basename, cache=False).posterior
    except FileNotFoundError:
        return None  # not launched, or not downloaded yet
    except Exception as _e:
        # unreadable traces (fit still running when downloaded)
        print(colored(f"  {basename}: unreadable traces ({type(_e).__name__}); "
                      f"fit likely still running, re-download when it finishes", "yellow"))
        return None
    _g = lambda v: _p[v].values.reshape(-1)
    _tm, _tt = _g("total_missed_cases"), _g("total_true_cases")
    _out = {
        "R_pre": _g("R_pre").mean(),
        "R_post": _g("R_post").mean(),
        "CFR_pre": _g("CFR_pre").mean() * 100,
        "CFR_post": _g("CFR_post").mean() * 100,
        "ua_cases": (_tm / _tt * 100).mean(),
        "ua_deaths": np.nan,
    }
    if "total_missed_deaths" in _p:  # imputed deaths among the missed cases
        _md = _g("total_missed_deaths")
        _out["ua_deaths"] = (_md / (_obs_deaths + _md) * 100).mean()
    return _out


_cells = {}
for _bn, _, _d in runs_grid:
    _s = _grid_stats(_bn)
    if _s is not None:
        _cells[(round(_d["epsilon"], 1), round(_d["theta"], 1))] = _s
print(colored(f"grid cells available: {len(_cells)}/{len(runs_grid)}", "cyan"))
if _cells and np.all(np.isnan([s["ua_deaths"] for s in _cells.values()])):
    print(
        colored(
            "  total_missed_deaths absent -> panel F blank",
            "yellow",
        )
    )

# Range of each quantity across the grid (reported in Appendix 3.3).
if _cells:
    _RANGE = [("R_pre", "Preintervention Re", 2), ("R_post", "Postintervention Re", 2),
              ("CFR_pre", "Preintervention CFR, %", 1), ("CFR_post", "Postintervention CFR, %", 1),
              ("ua_cases", "Overall under-ascertainment of cases, %", 1),
              ("ua_deaths", "Overall under-ascertainment of deaths, %", 1)]
    print(colored("\nrange across the (epsilon, theta) grid "
                  "[baseline = the epsilon = theta = 1 cell]:", "cyan"))
    _base = _cells.get((1.0, 1.0))
    for _k, _lab, _d in _RANGE:
        _v = np.array([c[_k] for c in _cells.values()], float)
        _v = _v[~np.isnan(_v)]
        if not len(_v):
            continue
        _b = "" if _base is None or np.isnan(_base[_k]) else f"   baseline {_base[_k]:.{_d}f}"
        print(f"  {_lab:42s} {_v.min():.{_d}f} to {_v.max():.{_d}f}{_b}")

# Plot the coarse subgrid if the full grid is incomplete.
_coarse_ax = sorted(_coarse)
if len(_cells) < len(runs_grid) and all(
    (e, t) in _cells for e in _coarse_ax for t in _coarse_ax
):
    _ex, _ty = _coarse_ax, _coarse_ax
    print(f"  plotting the complete coarse subgrid ({len(_ex)}x{len(_ty)})")
else:
    _ex, _ty = eps_values, theta_values


def _edges(v):
    """Cell boundaries for pcolormesh from centre coordinates."""
    v = np.asarray(v, float)
    _m = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - _m[0]], _m, [2 * v[-1] - _m[-1]]])


def _mat(key):
    M = np.full((len(_ty), len(_ex)), np.nan)
    for i, t in enumerate(_ty):
        for j, e in enumerate(_ex):
            s = _cells.get((e, t))
            if s is not None:
                M[i, j] = s[key]
    return M


def _nice_limits(vmin, vmax, n=5):
    """Round a data range outward to a readable step. Returns (lo, hi, step, decimals),
    so the colour scale covers the data exactly and the ticks land on round numbers."""
    _span = vmax - vmin
    if _span <= 0:
        return vmin - 0.5, vmax + 0.5, 1.0, 1
    _raw = _span / n
    _mag = 10.0 ** np.floor(np.log10(_raw))
    for _m in (1, 2, 2.5, 5, 10):
        _step = _m * _mag
        if _raw <= _step:
            break
    return (np.floor(vmin / _step) * _step, np.ceil(vmax / _step) * _step,
            _step, max(0, int(-np.floor(np.log10(_step)))))


# (key, title, panel letter, colour limits); None = data range rounded outward to a round step
_PANELS = [
    ("R_pre", r"Preintervention $\mathrm{R}_\text{eff}$", "A", None),
    ("R_post", r"Postintervention $\mathrm{R}_\text{eff}$", "B", None),
    ("CFR_pre", "Preintervention CFR, %", "C", None),
    ("CFR_post", "Postintervention CFR, %", "D", None),
    ("ua_cases", "Overall under-ascertainment\nof cases, %", "E", None),
    ("ua_deaths", "Overall under-ascertainment\nof deaths, %", "F", None),
]

# constrained_layout is on globally, so tune the layout engine (gridspec spacing is ignored)
# and use square cells
fig, axes = plt.subplots(3, 2, figsize=[5.4, 6.2])
fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.03, wspace=0.03)
_cmap = plt.get_cmap("Greys").copy()  # ColorBrewer Greys (low = light)
_cmap.set_bad("white")  # not-yet-fitted cells
_xe, _ye = _edges(_ex), _edges(_ty)

for _ax, (_key, _title, _lab, _lim) in zip(axes.ravel(), _PANELS):
    M = np.ma.masked_invalid(_mat(_key))
    if M.count() == 0:  # nothing to show
        _ax.text(
            0.5, 0.5, "not available", transform=_ax.transAxes,
            ha="center", va="center", fontsize=7, color="0.45",
        )
    # colour limits: rounded outward from the data unless pinned in _PANELS
    _vmin, _vmax, _step, _dec = _nice_limits(
        float(M.min()), float(M.max())) if M.count() else (0.0, 1.0, 0.25, 2)
    if _lim is not None:
        _vmin, _vmax = _lim
        _, _, _step, _dec = _nice_limits(_vmin, _vmax)
    # arrows on the bar flag any cell falling outside the limits (only possible if pinned)
    _ext = "neither"
    if M.count():
        _lo, _hi = float(M.min()) < _vmin, float(M.max()) > _vmax
        _ext = "both" if (_lo and _hi) else ("min" if _lo else ("max" if _hi else "neither"))
        if _ext != "neither":
            print(colored(
                f"  {_lab} ({_key}): data outside [{_vmin}, {_vmax}] -> observed "
                f"{float(M.min()):.2f} to {float(M.max()):.2f}", "yellow"))
    _mesh = _ax.pcolormesh(
        _xe, _ye, M, cmap=_cmap, norm=Normalize(_vmin, _vmax),
        shading="flat", edgecolors="face", linewidth=0,
    )
    _cb = fig.colorbar(_mesh, ax=_ax, fraction=0.046, pad=0.03, extend=_ext)
    _ticks = np.arange(_vmin, _vmax + _step / 2, _step)
    _cb.set_ticks(_ticks)
    _cb.ax.set_yticklabels([f"{v:.{_dec}f}" for v in _ticks])
    _cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    _cb.outline.set_linewidth(0.4)

    # un-fitted cells: hatch them, else white reads as "lowest value" on a Greys ramp
    _msk = np.ma.getmaskarray(M)
    if _msk.any():
        for _i in range(len(_ty)):
            for _j in range(len(_ex)):
                if _msk[_i, _j]:
                    _ax.add_patch(plt.Rectangle(
                        (_xe[_j], _ye[_i]), _xe[_j + 1] - _xe[_j], _ye[_i + 1] - _ye[_i],
                        facecolor="none", edgecolor="0.75", hatch="////", lw=0.0, zorder=3,
                    ))

    # The top-right cell (epsilon = theta = 1) is the baseline model (stated in the caption).
    _ax.set_aspect("equal")
    _ax.set_title(_title, fontsize=7.5, pad=3)
    _ax.set_xticks(_ex)
    _ax.set_yticks(_ty)
    _ax.set_xlabel(r"$\varepsilon$", fontsize=8, labelpad=1)
    _ax.set_ylabel(r"$\theta$", fontsize=8, labelpad=1)
    _ax.tick_params(labelsize=6, length=2, width=0.5)
    for _s in _ax.spines.values():
        _s.set_linewidth(0.5)
    _ax.text(
        -0.26, 1.16, _lab, transform=_ax.transAxes,
        fontsize=9, fontweight="bold", va="top", ha="left",
    )

plt.savefig(
    f"{figuresdir}/sensitivity_eps_theta_Bundibugyo.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
#

# %% Sensitivity: time from symptom onset to transmission (TOST)
# The baseline TOST is gamma with mean 4.8 d and SD 2.4 d (adapted from EBOV). Here the MEAN is
# varied (3.0, 4.5, 6.0 d) at the SAME SD. mean_ost/sd_ost are data in every daily model, so this
# reuses the baseline Stan file unchanged -- only Data.json differs.
_TOST_SRC = _SRC + "daily_gamma_step_semi_mechanistic_uniform_quasi-Binomial.stan"
tost_means = [3.0, 4.5, 6.0]

def _tost_bn(m):  # 4.5 -> "Rt_daily_gamma_BVD_tost45" (keeps dirnames free of dots)
    return f"Rt_daily_gamma_BVD_tost{int(round(m * 10)):02d}"

runs_tost = [
    (_tost_bn(_m), _TOST_SRC, {**_data_daily_gamma(7.71, 4.68), "mean_ost": float(_m)})
    for _m in tost_means
]

print(colored(f"\nTOST SWEEP: {len(runs_tost)} fits (SD held at 2.4 d)", "cyan"))
for _bn, _, _d in runs_tost:
    print(f"  {_bn:34s}  mean_ost={_d['mean_ost']:.1f}  sd_ost={_d['sd_ost']:.1f}")

# set to True to launch the fits, then to download them
if False:
    upload_and_launch(runs_tost)
if False:
    download_fits([b for b, _, _ in runs_tost])
#

# %% TOST sweep: comparison table (baseline 4.8 d included as the reference row)
_tost_spec = [(_m, _tost_bn(_m), "") for _m in tost_means]
_tost_spec.append((4.8, baseline_basename, " (baseline)"))
_tost_spec.sort(key=lambda r: r[0])

_rows_tost = []
for _m, _bn, _note in _tost_spec:
    _st = _model_row(_bn)
    if _st is not None:
        _rows_tost.append({"TOST mean, d": f"{_m:.1f}{_note}", **_st})

tost_table = pd.DataFrame(_rows_tost)
print(colored("\n=== TOST sensitivity (gamma TOST, SD 2.4 d; daily gamma BVD, "
              "semi-mechanistic, quasi-Binomial) ===", "cyan"))
print(tost_table.to_string(index=False) if len(tost_table)
      else colored("  (no TOST fits on disk yet)", "yellow"))
if len(tost_table):
    tost_table.to_csv(os.path.join(mainstandirname, "appendix_table_TOST.csv"), index=False)
    print(colored("wrote appendix_table_TOST.csv", "green"))
#
