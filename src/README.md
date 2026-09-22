# Scripts and Stan models

## Main script

[20260818 Bundibugyo 2007 outbreak.py](20260818%20Bundibugyo%202007%20outbreak.py) is a Python script organized in cells (`# %%`), meant to be run cell by cell in VS Code, Zed or Jupyter. It uses `pandas`, `numpy`, `scipy`, `matplotlib`, `cmdstanpy` and `arviz`. Its cells, in order:

1. **Machinery**: helper functions for posterior summaries (`get_stats`), a writer for the shell script that runs CmdStan (`write_bash_file`), and user-specific paths. The first line sources `psw.py` with the credentials of our computation server; this file is not shared.
2. **Loading data**: reads [data/wamala2010_weekly_cases.csv](../data/wamala2010_weekly_cases.csv), adds the total count and the week index, and sets the intervention week (48, week index 19, passed to Stan as `t_int`).
3. **Model grid and upload** (`upload_and_launch`): for every model variation, copies the Stan file, writes `Data.json` (see the data inputs below), obtains initial values with Pathfinder (`Inits.json`), writes `fit_bash.sh` and launches the fit on the remote server. The fits of the partial-ascertainment grid are launched in batches of 10 from a separate cell.
4. **Download and results**: fetches the traces (`trace-*.csv`) and sampler logs back, loads them with `cmdstanpy.from_csv` and `arviz`, and prints the posterior summary of the baseline model.
5. **Appendix Tables 1–5**: posterior means and 95% credible intervals for every model set, written to CSV files (Tables 1–4 also to one Excel workbook).
6. **Figures**: Figure 1 (boxplots), Appendix Figure 1 (three panels), the tau_rep sensitivity figures (Appendix Figures 2 and 3), the prior versus posterior comparison (Appendix Figure 5), the model fit and posterior predictive check (Appendix Figure 6), and the heatmaps over the partial-ascertainment grid (Appendix Figure 4). See [figures/README.md](../figures/README.md).

The fits were run on a remote Linux server with CmdStan (`STAN_THREADS=true`, 4 threads per chain): 5 chains, 2,500 warmup and 12,500 sampling iterations each, `adapt delta=0.99`, `max_depth=12`, with initial values obtained by Pathfinder. The traces are not included in this repository because of their size. The complete Stan output of all 147 fits, one directory per fit (named as described in the last section), is available in a shared [Dropbox folder](https://www.dropbox.com/scl/fo/fx43wb8v52uqlrb46p91d/AB2Vy0LIhPt8bENJw3TSqjU?rlkey=f6rz1fi4sav3zmy04izmxeevp&st=wtua2hy8&dl=0): each directory holds the posterior draws (`trace-1.csv` to `trace-5.csv`), the sampler logs (`output-1.txt` to `output-5.txt`), `Data.json`, `Inits.json`, the Stan file and, for most fits, the launch script `fit_bash.sh`. The folder also holds the CSV and Excel summary tables behind Appendix Tables 1 to 4.

## Stan models

All Stan files are in [stan_src](stan_src). Every model shares the same data (weekly counts of fatal and non-fatal cases, the intervention week `t_int`) and the same four estimated quantities:

| Parameter (Stan) | Quantity |
| --- | --- |
| `log_R` (ordered, `R_pre = exp(log_R[2])`, `R_post = exp(log_R[1])`) | Effective reproduction number before and after the intervention, `R_post < R_pre` |
| `p_check` | Ascertainment probability of cases before the intervention (under-ascertainment = `1 - p_check`); cases after the intervention are fully ascertained |
| `logit_CFR` (ordered, `CFR_pre = inv_logit(logit_CFR[2])`, `CFR_post = inv_logit(logit_CFR[1])`) | Case-fatality rate before and after the intervention, `CFR_post < CFR_pre` |
| `inv_sqrt_phi_rep` (`phi_rep = 1 / inv_sqrt_phi_rep^2`) | Overdispersion of the negative binomial observation model |

Priors (Appendix section 2.4.5): `log_R ~ normal(log 1.5, 1)`, `p_check ~ beta(6, 4)`, `logit_CFR ~ normal(0, 1.5)`, `inv_sqrt_phi_rep ~ exponential(1)`. The intervention is a step at `t_int`: the reproduction number switches from `R_pre` to `R_post`, ascertainment becomes complete, and the CFR is multiplied by `q = CFR_post / CFR_pre` (with the intermediate value `(1 + q) / 2` in the week before `t_int`). Three trailing weeks with zero cases are appended to the observed series. Deaths are assumed fully ascertained, so the weekly CFR likelihood uses the completed (observed plus missed) case counts, which are not integers. Generated quantities include the odds ratio of death `OR_exposure`, the weekly `Rt_infection` and `p_asc`, the sampled `missed_cases`, `total_missed_cases` and `total_true_cases`, the posterior predictive `case_pred` and its mean `case_pred_mean`, and the pointwise `log_lik`.

### Naming scheme

`Rt_Bundibugyo-{resolution}_{incubation}_{intervention}_{structure}_{CFR likelihood}.stan`

**Resolution** (Appendix 2.4, element 1)

* `daily`: daily renewal process on a 7-day grid within each week. The expected daily onsets are aggregated to the observed weekly totals through a moment-matched negative binomial (the weekly dispersion is `phi_rep * ec^2 / sum(daily^2)`).
* `weekly`: the renewal process runs directly on weekly intervals with the incubation period and TOST discretized to weeks (`w_max = 3`). Only the triangular incubation period was used at this resolution.

**Incubation period** (element 2). The time from onset to transmission (TOST) is a gamma distribution with mean 4.8 and SD 2.4 days in every model (data `mean_ost`, `sd_ost`, or the discretized `f_tost`).

* `gamma`: gamma incubation period parameterized by `mean_inc` and `sd_inc` (data). BVD: mean 7.71, SD 4.68 days (baseline). The same files were used with the EVD values (mean 11.4, SD 5.2 days, Appendix Table 4) and, for the baseline file, with the mean TOST set to 3.0, 4.5 or 6.0 days (Appendix Table 5).
* `Weibull`: Weibull incubation period given by `mean_inc` (7.4754 days) and the shape `param1_Weibull_inc` (1.8568).
* `lognormal`: lognormal incubation period given by `mean_inc` (8.1937 days) and `sd_inc` (6.0145 days); the model recomputes meanlog and sdlog.
* `triangular`: triangular incubation period with minimum 2, mode 7 and maximum 20 days, discretized in the script and passed as `f_inc` (daily, 28 bins) or as the weekly `f_inc` (3 bins).

**Intervention**

* `step`: the baseline step change described above.
* `ramp` (Appendix 3.2, Appendix Figures 2 and 3): the case ascertainment recovers towards the intervention as an exponential ramp with time constant `tau_rep` (data; 2, 4 or 8 weeks) instead of a step, so that retrospective case finding after week 48 reaches back in time with diminishing efficiency. `Rt_Bundibugyo-weekly_triangular.stan` is the weekly counterpart of this model (six fits in total).
* `partial_asc` (Appendix 2.4.4 and 3.3, Appendix Figure 4): two fixed data inputs relax the assumptions on ascertainment. `epsilon` is the completeness of case ascertainment after the intervention (`p_t = (1 - epsilon) * p + epsilon`), and `theta` the ascertainment of deaths among the missed cases (a fraction `1 - theta` of the deaths expected among the missed cases is added to the observed deaths). Both were varied over 0.1, 0.2, ..., 1.0 (100 fits); `epsilon = theta = 1` reproduces the baseline. This model also reports `total_missed_deaths`.

**Model structure** (elements 3 to 5)

* `semi_mechanistic_uniform` (baseline): the observed cases, completed for under-ascertainment, drive the force of infection (instantaneous reproduction number). The observed weekly counts are spread uniformly over the 7 days of the week.
* `semi_mechanistic_dirichlet`: as above, but the within-week allocation of the observed cases is a latent simplex of size 7 per week with a flat Dirichlet prior (parameter `case_within_week_allocation`).
* `latent_normal_vtm` (fully mechanistic, VTM): a state-space renewal model in which latent daily infections are drawn around the renewal mean with process noise whose variance is proportional to the mean, with the variance-to-mean ratio `d ~ exponential(1)`. A 14-day pre-observation seed block (2 weeks in the weekly models) holds a constant seed `j_seed ~ exponential(1 / tau_seed)` with `tau_seed ~ exponential(nu)` and `nu ~ half-normal(0, 2)`; innovations `eps` are standard normal (non-centred).
* `latent_normal_cv` (fully mechanistic, CV): the same model with the process SD proportional to the mean, with the coefficient of variation `cv ~ half-normal(0, 0.5)`.

At weekly resolution the semi-mechanistic model has no allocation suffix (`weekly_triangular_step_semi_mechanistic_*`).

**Case-fatality likelihood** (Appendix 2.4.1 and 2.4.3)

* `quasi-Binomial` (baseline): the binomial log-likelihood evaluated at the non-integer completed counts, with the log binomial coefficient written with log-gamma functions.
* `gamma_approx`: the moment-matched gamma approximation of the same likelihood (Appendix Table 2).

### Files

Daily models with a step intervention, 32 files (Appendix Table 1 with `quasi-Binomial`, Appendix Table 2 with `gamma_approx`; the `gamma` files also for Appendix Tables 4 and 5):

```
Rt_Bundibugyo-daily_{gamma|Weibull|lognormal|triangular}_step_{semi_mechanistic_uniform|semi_mechanistic_dirichlet|latent_normal_vtm|latent_normal_cv}_{quasi-Binomial|gamma_approx}.stan
```

The baseline model is `Rt_Bundibugyo-daily_gamma_step_semi_mechanistic_uniform_quasi-Binomial.stan` (Figure 1, Appendix Figures 1, 5 and 6, first row of Appendix Table 1).

Weekly models with a step intervention, 6 files (Appendix Table 3 used the `quasi-Binomial` ones; the `gamma_approx` versions are not reported in the paper):

```
Rt_Bundibugyo-weekly_triangular_step_{semi_mechanistic|latent_normal_vtm|latent_normal_cv}_{quasi-Binomial|gamma_approx}.stan
```

Sensitivity models, 3 files:

| File | Analysis |
| --- | --- |
| `Rt_Bundibugyo-daily_gamma_ramp_semi_mechanistic_uniform_quasi-Binomial.stan` | Time-varied ascertainment, `tau_rep` = 2, 4, 8 weeks (Appendix Figures 2 and 3) |
| `Rt_Bundibugyo-weekly_triangular.stan` | Weekly version of the ramp model (gamma-approximation CFR likelihood, no `log_lik`) |
| `Rt_Bundibugyo-daily_gamma_partial_asc_semi_mechanistic_uniform_quasi-Binomial.stan` | Partial ascertainment of cases and deaths, 10 x 10 grid over `epsilon` and `theta` (Appendix Figure 4) |

### Data inputs

The script builds the data for each model (`_data_weekly`, `_data_daily_tri`, `_data_daily_gamma`, `_data_daily_weibull`) and writes it to `Data.json`.

| Model family | Data in addition to `W`, `cases_fatal`, `cases_non_fatal`, `t_int` |
| --- | --- |
| `weekly_*` | `w_max = 3`, weekly `f_inc` (3 bins) and `f_tost` (4 bins, from week 0); `tau_rep` for the ramp model |
| `daily_triangular_*` | `d_max = 28`, `d_ost_max = 21`, daily `f_inc` and `f_tost` |
| `daily_gamma_*`, `daily_lognormal_*` | `d_inc_max = 28`, `d_ost_max = 21`, `mean_inc`, `sd_inc`, `mean_ost = 4.8`, `sd_ost = 2.4`; plus `tau_rep` (ramp) or `epsilon`, `theta` (partial ascertainment) |
| `daily_Weibull_*` | `d_inc_max = 28`, `d_ost_max = 21`, `mean_inc`, `param1_Weibull_inc`, `mean_ost = 4.8`, `sd_ost = 2.4` |

### Fit names used in the script

Each fit lives in a directory named after the model (these are the directory names in the Dropbox folder), for example `Rt_daily_gamma_BVD_semi_uniform_quasiBinomial` for the baseline. The tokens map to the file names as follows: `gamma_BVD`, `gamma_EVD`, `Weibull_BVD`, `lognormal_BVD` and `triangular` give the incubation period (and its parameter set); `semi_uniform`, `semi_dirichlet`, `fully_vtm` and `fully_cv` stand for `semi_mechanistic_uniform`, `semi_mechanistic_dirichlet`, `latent_normal_vtm` and `latent_normal_cv`; the suffix `_quasiBinomial` selects the `quasi-Binomial` file and its absence the `gamma_approx` file; `tau_{2,4,8}`, `partial_e{01..10}_t{01..10}` and `tost{30,45,60}` mark the sensitivity fits.
