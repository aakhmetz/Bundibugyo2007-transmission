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
    vector[w_max] f_inc_rev = reverse(f_inc);
    vector[w_max + 1] f_tost_rev = reverse(f_tost);

    array[W + 3] int<lower=0> cases;
    for (week in 1 : W)
        cases[week] = cases_fatal[week] + cases_non_fatal[week];
    cases[W + 1 : W + 3] = rep_array(0, 3);
}

parameters {
    // reproduction numbers
    ordered[2] log_R; // log_R[2] = log_R_pre, log_R[1] = log_R_post

    real<lower = 0> inv_sqrt_phi_rep;

    real<lower = 0, upper = 1> p_check;

    ordered[2] logit_CFR; // CFR_pre = CFR[2], CFR_post = CFR[1]

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
    for (week in 1 : W + 3) {
        Rt_infection[week] = t_int > week ? R_pre : R_post;
        p_asc[week] = t_int > week ? p_check : 1.0;
    }
}

model {
    log_R[2] ~ normal(log(1.5), 1.0);
    log_R[1] ~ normal(log(1.5), 1.0);
    inv_sqrt_phi_rep ~ exponential(1);

    p_check ~ beta(6, 4);

    logit_CFR ~ normal(0, 1.5);

    vector[W + 3] infections, expected_cases;
    {
        vector[W + 3] cases_asc = to_vector(cases);

        real conv_tost;
        for (week in 1 : W + 3) {
            expected_cases[week] = (week == 1) ? 0.0 : dot_product(tail(head(infections, week - 1), min(week - 1, w_max)), tail(f_inc_rev, min(week - 1, w_max)));
            cases_asc[week] += (1 - p_asc[week]) * expected_cases[week];
            conv_tost = dot_product(tail(head(cases_asc, week), min(week, w_max + 1)), tail(f_tost_rev, min(week, w_max + 1)));
            infections[week] = Rt_infection[week] * conv_tost;

            if (week <= W) {
                real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
                real prob_death = q_week * CFR_pre;
                int deaths = cases_fatal[week];
                real survived  = cases_asc[week] - deaths;
                if (deaths == 0)
                    target += cases_asc[week] * log1m(prob_death);
                else if (survived > 0)
                    target += gamma_lupdf(survived | deaths * (1 - prob_death), prob_death);
            }
        }

        target += neg_binomial_2_lupmf(cases[2 : (W + 3)] | tail(p_asc .* expected_cases, W + 2), phi_rep); // week 1 (index case) is excluded from the likelihood
    }
}

generated quantities {
    real OR_exposure = CFR_post / CFR_pre * (1 - CFR_pre) / (1 - CFR_post);

    // Missed cases per week are sampled from the same negative binomial observation process,
    // missed_x ~ NegBin2((1 - p_asc_x) * E[c_x], phi_rep); the dynamics use the deterministic mean.
    array[W] int missed_cases;             // sampled missed cases by week
    array[W] int cases_true;               // observed + missed, by week
    int total_missed_cases;                // sum of sampled missed cases
    int total_true_cases;                  // observed + total missed
    vector[W] expected_missed_cases;       // mean missed cases (for reference)
    vector[W] case_pred_mean;              // p_asc * E[c_x]: predicted reported-case mean
    array[W] int case_pred;                // posterior predictive reported cases
    vector[(W + 2) + W] log_lik;           // pointwise log-likelihood for LOO
    {
        vector[W + 3] infections, expected_cases;
        vector[W + 3] cases_asc = to_vector(cases);
        real conv_tost, mu_missed;
        for (week in 1 : W + 3) {
            expected_cases[week] = (week == 1) ? 0.0
                : dot_product(tail(head(infections, week - 1), min(week - 1, w_max)),
                              tail(f_inc_rev, min(week - 1, w_max)));
            mu_missed = (1 - p_asc[week]) * expected_cases[week];
            if (week <= W) {
                expected_missed_cases[week] = mu_missed;
                missed_cases[week] = mu_missed > 1e-9 ? neg_binomial_2_rng(mu_missed, phi_rep) : 0;
                cases_true[week]   = cases[week] + missed_cases[week];
                case_pred_mean[week] = p_asc[week] * expected_cases[week];
                case_pred[week] = case_pred_mean[week] > 1e-9
                    ? neg_binomial_2_rng(case_pred_mean[week], phi_rep) : 0;
            }
            cases_asc[week]   += mu_missed;   // deterministic mean drives the dynamics
            conv_tost = dot_product(tail(head(cases_asc, week), min(week, w_max + 1)),
                                    tail(f_tost_rev, min(week, w_max + 1)));
            infections[week] = Rt_infection[week] * conv_tost;
        }
        for (week in 2 : W + 3)
            log_lik[week - 1] = neg_binomial_2_lpmf(cases[week] | fmax(p_asc[week] * expected_cases[week], 1e-12), phi_rep);
        for (week in 1 : W) {
            real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
            real prob_death = q_week * CFR_pre;
            int deaths = cases_fatal[week];
            real survived = cases_asc[week] - deaths;
            log_lik[W + 2 + week] = (deaths == 0) ? cases_asc[week] * log1m(prob_death)
                : (survived > 0 ? gamma_lpdf(survived | deaths * (1 - prob_death), prob_death) : deaths * log(prob_death));
        }
        total_missed_cases = sum(missed_cases);
        total_true_cases   = sum(cases) + total_missed_cases;
    }
}
