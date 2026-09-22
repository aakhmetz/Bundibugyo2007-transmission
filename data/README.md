# Data

Individual-level data for the 2007 Bundibugyo virus outbreak are not publicly available. The weekly incidence was digitized from Figure 2 of Wamala JF, Lukwago L, Malimbo M, et al. "Ebola hemorrhagic fever associated with novel virus strain, Uganda, 2007–2008" *Emerg Infect Dis* 2010;16(7):1087–92 (doi:[10.3201/eid1607.091525](https://doi.org/10.3201/eid1607.091525)). That figure shows probable and confirmed cases by week of symptom onset, stratified by outcome (fatal cases as white bars, non-fatal cases as black bars). The bars were read with a sub-pixel fit and sum exactly to 110 cases (37 fatal, 73 non-fatal). The 116 cases quoted in the source include 6 cases (2 deaths) without a known week of onset, which are therefore absent from the epidemic curve.

## Bundibugyo_2007_Uganda_epicurve.xlsx

The digitized series as extracted, from week 28 of 2007 to week 2 of 2008 (28 rows). Sheet `epicurve` holds the counts and sheet `notes` the provenance notes above.

| Column | Description |
| --- | --- |
| `year` | Year of the epidemiologic week |
| `epiweek` | ISO 8601 week number of the week of symptom onset |
| `week_start` | First day (Monday) of that week |
| `cases` | Probable and confirmed cases with onset in the week (`deaths` + `survived`) |
| `deaths` | Fatal cases (white bars in the source figure) |
| `survived` | Non-fatal cases (black bars in the source figure) |

## wamala2010_weekly_cases.csv

The file read by the main script. It is the same series restricted to the weeks from the first case (week 30 of 2007) to the last week with reported cases (week 52 of 2007), 23 rows.

| Column | Description |
| --- | --- |
| `week` | Epidemiologic week of 2007 of symptom onset (`epiweek` above) |
| `count_fatal` | Fatal cases (`deaths` above) |
| `count_non_fatal` | Non-fatal cases (`survived` above) |

The script derives `count_total` (their sum) and `week_index` (1 for week 30, 23 for week 52). The intervention week (48, `week_index` 19) is set in the script as `intervention_effect_week` and enters the Stan models as `t_int`. Three weeks with zero cases are appended after week 52 inside the Stan models, so that the end of the outbreak contributes to the likelihood.
