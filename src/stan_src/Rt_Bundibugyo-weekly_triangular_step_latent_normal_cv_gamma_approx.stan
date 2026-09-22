// Weekly state-space renewal model: latent weekly infections ~ Normal(R_t*force, cv*R_t*force),
// floored above 0 and sampled non-centred (CV parameterisation, epidemia fixed_vtm = FALSE),
// preceded by a 2-week seed block of latent infections (constant seed).
data {
    int<lower = 1> W; // number of weeks
    array[W] int<lower = 0> cases_fatal, cases_non_fatal; // weekly counts

    real<lower = 1, upper = W> t_int; // first week (index) of the post-intervention period

    // discrete incper and TOST
    int<lower = 1> w_max; // maximum number of weeks in the discretized distributions
    vector<lower = 0, upper = 1>[w_max] f_inc; // discretized incubation period distribution (starts from week 1)
    vector<lower = 0, upper = 1>[w_max + 1] f_tost; // discretized TOST distribution (starts from week 0)
}

transformed data {
    int n_seed = 2;                        // pre-observation seed block (weeks, ~14 days)
    int Tw = n_seed + (W + 3);             // seed block + observation timeline (weeks)

    vector[w_max] f_inc_rev = reverse(f_inc);
    vector[w_max + 1] f_tost_rev = reverse(f_tost);

    array[W + 3] int<lower=0> cases;
    for (week in 1 : W)
        cases[week] = cases_fatal[week] + cases_non_fatal[week];
    cases[W + 1 : W + 3] = rep_array(0, 3);
}

parameters {
    ordered[2] log_R; // log_R[2] = log_R_pre, log_R[1] = log_R_post
    real<lower = 0> inv_sqrt_phi_rep;
    real<lower = 0, upper = 1> p_check;
    ordered[2] logit_CFR; // CFR_pre = CFR[2], CFR_post = CFR[1]

    real<lower = 0> nu;                    // hyper-rate for tau (estimated; half-normal prior)
    real<lower = 0> tau_seed;              // mean of the constant weekly seed
    real<lower = 0> j_seed;                // constant weekly seed over the pre-observation window
    real<lower = 0> cv;                    // coefficient of variation of latent infections (fixed_vtm = FALSE)
    vector[W + 3] eps;                     // standardised innovations, one per renewal week
}

transformed parameters {
    real phi_rep = inv_square(inv_sqrt_phi_rep);

    real R_pre = exp(log_R[2]),
        R_post = exp(log_R[1]),
        eta = R_post / R_pre,
        CFR_post = inv_logit(logit_CFR[1]),
        CFR_pre = inv_logit(logit_CFR[2]),
        q = CFR_post / CFR_pre;


    vector[W + 3] Rt_infection, p_asc;
    vector[Tw] infections, onsets;
    real mean_inf, force;
    for (week in 1 : W + 3) {
        Rt_infection[week] = t_int > week ? R_pre : R_post;
        p_asc[week] = t_int > week ? p_check : 1.0;
    }
    for (wk in 1 : Tw) {
        onsets[wk] = (wk == 1) ? 0.0
            : dot_product(tail(head(infections, wk - 1), min(wk - 1, w_max)),
                          tail(f_inc_rev, min(wk - 1, w_max)));
        if (wk <= n_seed) {
            infections[wk] = j_seed;                                // latent seed (pre-observation)
        } else {
            int ow = wk - n_seed;                            // observation week (1..W+3)
            force = dot_product(tail(head(onsets, wk), min(wk, w_max + 1)),
                                tail(f_tost_rev, min(wk, w_max + 1)));
            mean_inf = Rt_infection[ow] * force;
            infections[wk] = fmax(mean_inf * (1 + cv * eps[ow]), 1e-12);
        }
    }
    vector[W + 3] onsets_obs = onsets[n_seed + 1 : Tw];       // onsets in the observation window
}

model {
    log_R[2] ~ normal(log(1.5), 1.0);
    log_R[1] ~ normal(log(1.5), 1.0);
    inv_sqrt_phi_rep ~ exponential(1);
    p_check ~ beta(6, 4);
    logit_CFR ~ normal(0, 1.5);

    nu ~ normal(0, 2);                    // half-normal (lower = 0) on the hyper-rate
    tau_seed ~ exponential(nu);           // epidemia hierarchy: tau ~ Exp(nu)
    j_seed ~ exponential(inv(tau_seed));  // constant seed j ~ Exp(1/tau), mean tau
    cv ~ normal(0, 0.5);            // half-normal (lower = 0) on the coefficient of variation
    eps ~ std_normal();

    {
        vector[W + 3] mu_obs;
        for (week in 1 : W + 3)
            mu_obs[week] = fmax(p_asc[week] * onsets_obs[week], 1e-12);
        target += neg_binomial_2_lupmf(cases[2 : (W + 3)] | tail(mu_obs, W + 2), phi_rep);

        for (week in 1 : W) {
            real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
            real prob_death = q_week * CFR_pre;
            int deaths = cases_fatal[week];
            real c_asc = cases[week] + (1 - p_asc[week]) * onsets_obs[week];
            real survived = c_asc - deaths;
            if (deaths == 0)
                target += c_asc * log1m(prob_death);
            else if (survived > 0)
                target += gamma_lupdf(survived | deaths * (1 - prob_death), prob_death);
        }
    }
}

generated quantities {
    real OR_exposure = CFR_post / CFR_pre * (1 - CFR_pre) / (1 - CFR_post);
    real total_seed = n_seed * j_seed;   // total pre-observation seeded infections (diagnostic)

    array[W] int missed_cases, cases_true, case_pred;
    int total_missed_cases, total_true_cases;
    vector[W] expected_missed_cases, case_pred_mean;
    // full pointwise log-likelihood for LOO: weekly case counts (2..W+3) then CFR (1..W)
    vector[(W + 2) + W] log_lik;
    for (week in 2 : W + 3)
        log_lik[week - 1] = neg_binomial_2_lpmf(cases[week] | fmax(p_asc[week] * onsets_obs[week], 1e-12), phi_rep);
    for (week in 1 : W) {
        real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
        real prob_death = q_week * CFR_pre;
        int deaths = cases_fatal[week];
        real c_asc = cases[week] + (1 - p_asc[week]) * onsets_obs[week];
        real survived = c_asc - deaths;
        log_lik[W + 2 + week] = (deaths == 0) ? c_asc * log1m(prob_death)
            : (survived > 0 ? gamma_lpdf(survived | deaths * (1 - prob_death), prob_death)
                            : deaths * log(prob_death));
    }
    for (week in 1 : W) {
        real mu_missed = (1 - p_asc[week]) * onsets_obs[week];
        expected_missed_cases[week] = mu_missed;
        case_pred_mean[week] = p_asc[week] * onsets_obs[week];
        missed_cases[week] = mu_missed > 1e-9 ? neg_binomial_2_rng(mu_missed, phi_rep) : 0;
        case_pred[week] = case_pred_mean[week] > 1e-9 ? neg_binomial_2_rng(case_pred_mean[week], phi_rep) : 0;
        cases_true[week] = cases[week] + missed_cases[week];
    }
    total_missed_cases = sum(missed_cases);
    total_true_cases   = sum(cases) + total_missed_cases;
}
