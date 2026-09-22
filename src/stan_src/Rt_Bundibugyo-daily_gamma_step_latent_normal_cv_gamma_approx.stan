// Daily state-space renewal model, gamma incubation period: latent daily infections with coefficient-of-variation (CV)
// process noise, preceded by a 14-day seed block of latent infections (constant seed, non-centred),
// following the epidemia framework. Daily onsets are aggregated to weekly counts (moment-matched NB).
data {
    int<lower = 1> W;
    array[W] int<lower = 0> cases_fatal, cases_non_fatal;

    real<lower = 1, upper = W> t_int;

    int<lower = 1> d_inc_max, d_ost_max;
    real<lower = 0> mean_inc, sd_inc, mean_ost, sd_ost;
}

transformed data {
    int n_seed = 14;                       // pre-observation seed block (days, ~1 Ebola generation)
    int T = n_seed + 7 * (W + 3);          // seed block + observation timeline (days)

    real param1_inc = square(mean_inc / sd_inc), param2_inc = mean_inc / square(sd_inc);
    vector[d_inc_max] f_rev;
    {
        vector[d_inc_max + 1] res;
        for (k in 1 : d_inc_max + 1)
            res[k] = gamma_cdf(k - 0.5 | param1_inc, param2_inc);
        vector[d_inc_max] f = tail(res, d_inc_max) - head(res, d_inc_max);
        f_rev = reverse(f / sum(f));
    }
    real param1_ost = square(mean_ost / sd_ost), param2_ost = mean_ost / square(sd_ost);
    vector[d_ost_max + 1] lambda_rev;
    {
        vector[d_ost_max + 1] res;
        for (k in 1 : d_ost_max + 1)
            res[k] = gamma_cdf(k - 0.5 | param1_ost, param2_ost);
        vector[d_ost_max + 1] lambda = append_row(res[1] / res[d_ost_max + 1],
            (tail(res, d_ost_max) - head(res, d_ost_max)) / res[d_ost_max + 1]);
        lambda_rev = reverse(lambda);
    }
    array[W + 3] int<lower=0> cases;
    for (week in 1 : W)
        cases[week] = cases_fatal[week] + cases_non_fatal[week];
    cases[W + 1 : W + 3] = rep_array(0, 3);
}

parameters {
    ordered[2] log_R;
    real<lower = 0> inv_sqrt_phi_rep;
    real<lower = 0, upper = 1> p_check;
    ordered[2] logit_CFR;

    real<lower = 0> nu;                    // hyper-rate for tau (estimated; half-normal prior)
    real<lower = 0> tau_seed;              // mean of the constant daily seed
    real<lower = 0> j_seed;                // constant daily seed over the pre-observation window
    real<lower = 0> cv;                    // coefficient of variation of latent daily infections
    vector[7 * (W + 3)] eps;               // standardised daily innovations, one per renewal day
}

transformed parameters {
    real phi_rep = inv_square(inv_sqrt_phi_rep);

    real R_pre = exp(log_R[2]),
        R_post = exp(log_R[1]),
        eta = R_post / R_pre,
        CFR_post = inv_logit(logit_CFR[1]),
        CFR_pre = inv_logit(logit_CFR[2]),
        q = CFR_post / CFR_pre;

    vector[W + 3] Rt_infection, p_asc, E_week, phi_rep_weekly;
    vector[T] infections, onsets;
    real mean_inf, force;
    for (week in 1 : W + 3) {
        Rt_infection[week] = t_int > week ? R_pre : R_post;
        p_asc[week] = t_int > week ? p_check : 1.0;
    }
    for (day in 1 : T) {
        onsets[day] = (day == 1) ? 0.0
            : dot_product(tail(head(infections, day - 1), min(day - 1, d_inc_max)),
                          tail(f_rev, min(day - 1, d_inc_max)));
        if (day <= n_seed) {
            infections[day] = j_seed;                        // constant latent seed (pre-observation)
        } else {
            int obs_day = day - n_seed;                      // 1 .. 7*(W+3)
            int week = (obs_day - 1) %/% 7 + 1;
            force = dot_product(tail(head(onsets, day), min(day, d_ost_max + 1)),
                                tail(lambda_rev, min(day, d_ost_max + 1)));
            mean_inf = Rt_infection[week] * force;
            infections[day] = fmax(mean_inf * (1 + cv * eps[obs_day]), 1e-12);
        }
    }
    for (week in 1 : W + 3) {
        int base = n_seed + 7 * (week - 1);                  // shift past the seed block
        real ec = sum(onsets[base + 1 : base + 7]);
        real s2 = sum(square(onsets[base + 1 : base + 7]));
        E_week[week] = ec;
        phi_rep_weekly[week] = s2 > 0 ? phi_rep * square(ec) / s2 : phi_rep;  // moment-matched weekly NB
    }
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
    cv ~ normal(0, 0.5);
    eps ~ std_normal();

    {
        vector[W + 3] mu_obs;
        for (week in 1 : W + 3)
            mu_obs[week] = fmax(p_asc[week] * E_week[week], 1e-12);
        target += neg_binomial_2_lupmf(cases[2 : (W + 3)] | tail(mu_obs, W + 2), tail(phi_rep_weekly, W + 2));

        for (week in 1 : W) {
            real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
            real prob_death = q_week * CFR_pre;
            int deaths = cases_fatal[week];
            real c_asc = cases[week] + (1 - p_asc[week]) * E_week[week];
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
        log_lik[week - 1] = neg_binomial_2_lpmf(cases[week] | fmax(p_asc[week] * E_week[week], 1e-12), phi_rep_weekly[week]);
    for (week in 1 : W) {
        real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
        real prob_death = q_week * CFR_pre;
        int deaths = cases_fatal[week];
        real c_asc = cases[week] + (1 - p_asc[week]) * E_week[week];
        real survived = c_asc - deaths;
        log_lik[W + 2 + week] = (deaths == 0) ? c_asc * log1m(prob_death)
            : (survived > 0 ? gamma_lpdf(survived | deaths * (1 - prob_death), prob_death)
                            : deaths * log(prob_death));
    }
    for (week in 1 : W) {
        real mu_missed = (1 - p_asc[week]) * E_week[week];
        expected_missed_cases[week] = mu_missed;
        case_pred_mean[week] = p_asc[week] * E_week[week];
        missed_cases[week] = mu_missed > 1e-9 ? neg_binomial_2_rng(mu_missed, phi_rep_weekly[week]) : 0;
        case_pred[week] = case_pred_mean[week] > 1e-9 ? neg_binomial_2_rng(case_pred_mean[week], phi_rep_weekly[week]) : 0;
        cases_true[week] = cases[week] + missed_cases[week];
    }
    total_missed_cases = sum(missed_cases);
    total_true_cases   = sum(cases) + total_missed_cases;
}
