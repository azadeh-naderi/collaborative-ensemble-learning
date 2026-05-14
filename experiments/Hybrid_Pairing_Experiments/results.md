## results 

| Policy           | avg_ens_acc            | avg_learner_acc        | avg_ens_conf              | mean_pairwise_disagreement | distilled_student_acc |
|------------------|------------------------|------------------------|----------------------------|------------------------------|----------------------------------|
| AccDiff          | 83.11   | 81.24   | 0.809    | 0.117     | 83.85             |
| MWM_AccDiff      | **84.55** | **82.57** | **0.820** | 0.120      | **84.56**         |
| MWM_AccDiff - Remove Oracle          | 76.45   | 75.52   | 0.79    | 0.104     | 77.49             |
| Start AccDiff - Switch to MWM_AccDiff          | 84.13   | 82.3   | 0.82    | 0.117     | 83.38             |
