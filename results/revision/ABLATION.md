# Feature and model ablation

Same 24,000 reads (12,000 human, 12,000 pathogen) and same flow-cell-held-out folds as diagnostics stage H1. Each human flow cell is tested once, with a species-stratified third of the pathogen reads. Values are mean +/- sd over the 3 folds. Raw per-fold numbers with bootstrap CIs are in `ablation.json`; script is `ablation.py`.

| Model | Features | AUC | Accuracy |
|---|---|---|---|
| MLP on raw normalised window | 4000 | 0.539 +/- 0.002 | 0.511 +/- 0.014 |
| MLP, 70 statistical features only | 70 | 0.985 +/- 0.002 | 0.936 +/- 0.009 |
| MLP, 210 filter features only | 210 | 0.892 +/- 0.004 | 0.805 +/- 0.007 |
| **MLP, all 280 (FreeSolo)** | 280 | 0.990 +/- 0.002 | 0.951 +/- 0.008 |
| Logistic regression, all 280 | 280 | 0.989 +/- 0.001 | 0.960 +/- 0.004 |
| Full minus edge filters | 224 | 0.988 +/- 0.002 | 0.942 +/- 0.011 |
| Full minus LoG filters | 238 | 0.990 +/- 0.001 | 0.946 +/- 0.006 |
| Full minus sinusoids | 210 | 0.991 +/- 0.002 | 0.951 +/- 0.010 |
| Full minus moving averages | 238 | 0.988 +/- 0.002 | 0.943 +/- 0.008 |

## Reading
- Feature extraction matters: the same MLP on the raw window is at chance.
- The filters alone are weaker than 70 simple statistics. Adding them to the statistics gives +0.005 AUC (0.985 to 0.990) and +1.5 points accuracy. Small, consistent across folds, but a modest gain.
- Logistic regression on the same 280 features matches the MLP. The MLP adds nothing measurable over a linear model here.
- No single filter family is needed: dropping any one changes AUC by <= 0.002 (within fold noise). Sinusoids are not helping.
- These are held-out-flow-cell results, but the human and pathogen classes still differ in instrument, kit and run, so none of this shows the features detect species rather than run chemistry (see the confound diagnostics).
