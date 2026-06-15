# Minima Sharpness Comparison (job 1070319)

Sharpness score = mean test-accuracy drop (%) under relative Gaussian weight
perturbation across sigma in {0.01, 0.02, 0.05, 0.1, 0.2}, averaged over all
checkpoints for each method (lower = flatter / wider minimum). CelNet's
oracle model (idx=0) is excluded.

| Method | Score | Test Acc |
|---|---|---|
| CelNet | 1.34 | ~83.0% |
| KD | 2.37 | ~87.2% |
| CE_24epochs | 2.52 | ~80.6% |
| CE | 2.87 | ~86.6% |

CelNet converges to the flattest minimum despite mid-range accuracy. Notably,
CE_24epochs (fewer training epochs, lowest accuracy) is still sharper than
CelNet, indicating CelNet's flatness is not merely an artifact of less
training.

See `sharpness_comparison.png`, `flatness_summary.csv`, and
`sharpness_score.csv` for full results.
