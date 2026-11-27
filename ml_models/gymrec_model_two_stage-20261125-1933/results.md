## Two-stage model

Each stage chosen on its validation score.

| Stage | Exercises | Chosen model | Val acc | Val F1 | Test acc | Test F1 |
|---|---|---|---|---|---|---|
| router | groups: arms, core, legs, press, pull | XGBoost | 0.982 | 0.982 | 0.818 | 0.809 |
| arms | bicep_curl, hammer_curl, lateral_raise, tricep_pushdown | Random Forest | 0.920 | 0.945 | 0.707 | 0.726 |
| core | leg_raises, plank, russian_twist | LSTM | 1.000 | 1.000 | 0.788 | 0.778 |
| legs | deadlift, hip_thrust, leg_extension, romanian_deadlift, squat | 1D-CNN / TCN | 0.962 | 0.962 | 0.747 | 0.782 |
| press | bench_press, chest_fly_machine, decline_bench_press, incline_bench_press, push_up, shoulder_press, tricep_dips | 1D-CNN / TCN | 0.962 | 0.961 | 0.630 | 0.596 |
| pull | lat_pulldown, pull_up, t_bar_row | LSTM | 1.000 | 1.000 | 0.920 | 0.921 |

Specialist scores assume the router picked the right group.

## Whole chain vs single model

| Approach | Val acc | Val F1 | Test acc | Test F1 |
|---|---|---|---|---|
| **Two-stage** | 0.956 | 0.953 | 0.621 | 0.621 |
| Single model | (not trained on these classes) | | | |

All stages, every candidate (validation):

- router: Random Forest 0.964, XGBoost 0.982, LSTM 0.956, GRU 0.950, 1D-CNN / TCN 0.960
- arms: Random Forest 0.945, XGBoost 0.943, LSTM 0.850, GRU 0.902, 1D-CNN / TCN 0.909
- core: Random Forest 1.000, XGBoost 1.000, LSTM 1.000, GRU 0.995, 1D-CNN / TCN 1.000
- legs: Random Forest 0.919, XGBoost 0.920, LSTM 0.943, GRU 0.952, 1D-CNN / TCN 0.962
- press: Random Forest 0.909, XGBoost 0.929, LSTM 0.955, GRU 0.944, 1D-CNN / TCN 0.961
- pull: Random Forest 0.993, XGBoost 1.000, LSTM 1.000, GRU 1.000, 1D-CNN / TCN 1.000

Most confused on test (true → predicted): deadlift → romanian_deadlift (31), incline_bench_press → decline_bench_press (27), romanian_deadlift → deadlift (26), shoulder_press → lat_pulldown (23), hammer_curl → bicep_curl (16)
