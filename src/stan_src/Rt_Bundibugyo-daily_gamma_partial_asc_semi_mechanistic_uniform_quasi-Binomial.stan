// Partial ascertainment of cases and deaths (Appendix 2.4.4 and 3.3). Two quantities fixed as data:
//   epsilon: case ascertainment after the intervention, p_t = (1 - epsilon) * p + epsilon
//   theta:   ascertainment of deaths among the missed cases; the completed weekly deaths are
//            d~_w = d_w + (1 - theta) * q_w * CFR * (expected missed cases in week w)
// epsilon = theta = 1 reproduces the baseline model (daily gamma incubation period,
// semi-mechanistic with uniform allocation, quasi-Binomial CFR).
// Case-fatality likelihood: binomial evaluated at the non-integer completed counts (log-gamma
// binomial coefficient). Otherwise identical to the corresponding _gamma_approx model.
data {
    int<lower = 1> W; // number of weeks
    array[W] int<lower = 0> cases_fatal, cases_non_fatal; // weekly counts

    real<lower = 1, upper = W> t_int;

    // discrete incper and TOST
    int<lower = 1> d_inc_max, d_ost_max; // maximum number of days in the discretized distributions
    real<lower = 0> mean_inc, sd_inc,
        mean_ost, sd_ost;

    // partial-ascertainment quantities (fixed as data; 1 = baseline)
    real<lower = 0, upper = 1> epsilon;   // case ascertainment after the intervention:
                                          //   p_t = (1 - epsilon) * p + epsilon
    real<lower = 0, upper = 1> theta;     // ascertainment of deaths among the missed cases:
                                          //   d~_w = d_w + (1 - theta) * q_w * CFR * (missed cases)
}

transformed data {
    // parameters of the gamma distribution for the incubation period
    real param1_inc = square(mean_inc / sd_inc),
        param2_inc = mean_inc / square(sd_inc);
    vector[d_inc_max] f_rev;
    {
        vector[d_inc_max + 1] res;
        for (k in 1 : d_inc_max + 1)
            res[k] = gamma_cdf(k - 0.5 | param1_inc, param2_inc);
        vector[d_inc_max] f = tail(res, d_inc_max) - head(res, d_inc_max);
        f_rev = reverse(f / sum(f));
    }

    // parameters of the gamma distribution for the TOST
    real param1_ost = square(mean_ost / sd_ost),
        param2_ost = mean_ost / square(sd_ost);
    vector[d_ost_max + 1] lambda_rev;
    {
        vector[d_ost_max + 1] res;
        for (k in 1 : d_ost_max + 1)
            res[k] = gamma_cdf(k - 0.5 | param1_ost, param2_ost);
        vector[d_ost_max + 1] lambda = append_row(res[1] / res[d_ost_max + 1], (tail(res, d_ost_max) - head(res, d_ost_max)) / res[d_ost_max + 1]);
        lambda_rev = reverse(lambda);
    }

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
        p_asc[week] = t_int > week ? p_check : fma(1 - epsilon, p_check, epsilon);
    }
}

model {
    log_R[2] ~ normal(log(1.5), 1.0);
    log_R[1] ~ normal(log(1.5), 1.0);
    inv_sqrt_phi_rep ~ exponential(1);

    p_check ~ beta(6, 4);

    logit_CFR ~ normal(0, 1.5);

    vector[7 * (W + 3)] infections, expected_cases;
    {
        real conv_tost;
        vector[7 * (W + 3)] cases_asc;
        for (week in 1 : (W + 3))
            cases_asc[7 * (week - 1) + 1 : 7 * week] = rep_vector(cases[week] / 7., 7);
        for (day in 1 : 7 * (W + 3)) {
            int week = (day - 1) %/% 7 + 1;
            expected_cases[day] = (day == 1) ? 0.0 : dot_product(tail(head(infections, day - 1), min(day - 1, d_inc_max)), tail(f_rev, min(day - 1, d_inc_max)));
            cases_asc[day] += (1 - p_asc[week]) * expected_cases[day];
            conv_tost = dot_product(tail(head(cases_asc, day), min(day, d_ost_max + 1)), tail(lambda_rev, min(day, d_ost_max + 1)));
            infections[day] = Rt_infection[week] * conv_tost;
        }

        vector[W + 3] mu_week, phi_rep_weekly, expected_cases_weekly;
        for (week in 1 : W + 3) {
            real ec = sum(expected_cases[7 * (week - 1) + 1 : 7 * week]);
            real s2 = sum(square(expected_cases[7 * (week - 1) + 1 : 7 * week]));
            expected_cases_weekly[week] = ec;
            mu_week[week] = fmax(p_asc[week] * ec, 1e-12);
            phi_rep_weekly[week] = s2 > 0 ? phi_rep * square(ec) / s2 : phi_rep;
        }

        target += neg_binomial_2_lupmf(cases | mu_week, phi_rep_weekly);

        for (week in 1 : W) {
            real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
            real prob_death = q_week * CFR_pre;
            real missed = (1 - p_asc[week]) * expected_cases_weekly[week];        // unascertained cases
            real c_asc = cases[week] + missed;                                    // completed cases (binomial size)
            real deaths = cases_fatal[week] + (1 - theta) * prob_death * missed;  // completed deaths
            real survived = c_asc - deaths;
            if (deaths <= 0)
                target += c_asc * log1m(prob_death);
            else if (survived > 0)
                target += (lgamma(c_asc + 1) - lgamma(deaths + 1) - lgamma(survived + 1) + deaths * log(prob_death) + survived * log1m(prob_death));
        }
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
    vector[W] missed_deaths_wk;            // deaths imputed among the unascertained cases
    real total_missed_deaths;              // their total (0 when theta = 1)
    {
        vector[7 * (W + 3)] infections, expected_cases;
        vector[7 * (W + 3)] cases_asc;
        for (week in 1 : (W + 3))
            cases_asc[7 * (week - 1) + 1 : 7 * week] = rep_vector(cases[week] / 7., 7);
        real conv_tost;
        for (day in 1 : 7 * (W + 3)) {
            int week = (day - 1) %/% 7 + 1;
            expected_cases[day] = (day == 1) ? 0.0 : dot_product(tail(head(infections, day - 1), min(day - 1, d_inc_max)), tail(f_rev, min(day - 1, d_inc_max)));
            cases_asc[day] += (1 - p_asc[week]) * expected_cases[day];
            conv_tost = dot_product(tail(head(cases_asc, day), min(day, d_ost_max + 1)), tail(lambda_rev, min(day, d_ost_max + 1)));
            infections[day] = Rt_infection[week] * conv_tost;
        }

        vector[W + 3] expected_cases_weekly, phi_rep_weekly;
        for (week in 1 : W + 3) {
            real ec = sum(expected_cases[7 * (week - 1) + 1 : 7 * week]);
            real s2 = sum(square(expected_cases[7 * (week - 1) + 1 : 7 * week]));
            expected_cases_weekly[week] = ec;
            phi_rep_weekly[week] = s2 > 0 ? phi_rep * square(ec) / s2 : phi_rep;
        }
        for (week in 2 : W + 3)
            log_lik[week - 1] = neg_binomial_2_lpmf(cases[week] | fmax(p_asc[week] * expected_cases_weekly[week], 1e-12), phi_rep_weekly[week]);
        for (week in 1 : W) {
            real q_week = (week >= t_int) ? q : ((week == t_int - 1) ? (1 + q) / 2 : 1.0);
            real prob_death = q_week * CFR_pre;
            real missed = (1 - p_asc[week]) * expected_cases_weekly[week];
            real c_asc = cases[week] + missed;
            real deaths = cases_fatal[week] + (1 - theta) * prob_death * missed;
            real survived = c_asc - deaths;
            missed_deaths_wk[week] = (1 - theta) * prob_death * missed;
            log_lik[W + 2 + week] = (deaths <= 0) ? c_asc * log1m(prob_death)
                : (survived > 0 ? (lgamma(c_asc + 1) - lgamma(deaths + 1) - lgamma(survived + 1) + deaths * log(prob_death) + survived * log1m(prob_death)) : deaths * log(prob_death));
        }

        for (week in 1 : W) {
            real mu_missed = (1 - p_asc[week]) * expected_cases_weekly[week];
            expected_missed_cases[week] = mu_missed;
            case_pred_mean[week] = p_asc[week] * expected_cases_weekly[week];
            missed_cases[week] = mu_missed > 1e-9 ? neg_binomial_2_rng(mu_missed, phi_rep_weekly[week]) : 0;
            case_pred[week] = case_pred_mean[week] > 1e-9 ? neg_binomial_2_rng(case_pred_mean[week], phi_rep_weekly[week]) : 0;
            cases_true[week] = cases[week] + missed_cases[week];
        }
        total_missed_deaths = sum(missed_deaths_wk);
        total_missed_cases = sum(missed_cases);
        total_true_cases   = sum(cases) + total_missed_cases;
    }
}
