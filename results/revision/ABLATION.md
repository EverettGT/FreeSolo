# Feature and classifier ablation

Question: which parts of FreeSolo are needed? In particular, do the hand-designed filters help, and does their design matter?

Data: the main sample of 12,000 human and 12,000 bacterial reads cached by `diagnostics.py`. Human test reads always come from a held-out flow cell.

## Classifier comparison (`ablation.py`, 3 flow-cell folds, `ablation.json`)

| Features | Logistic regression AUC | Logistic regression acc | MLP AUC | MLP acc |
| --- | --- | --- | --- | --- |
| Raw normalised window (4,000) | 0.505 ± 0.003 | 0.494 ± 0.005 | 0.539 ± 0.002 | 0.511 ± 0.014 |
| 70 statistical | | | 0.985 ± 0.002 | 0.936 ± 0.009 |
| 210 filter | | | 0.892 ± 0.004 | 0.805 ± 0.007 |
| All 280 | 0.989 ± 0.001 | 0.960 ± 0.004 | 0.990 ± 0.002 | 0.951 ± 0.008 |

MLP family drops (all 280 minus one family): edges 0.988 ± 0.002, log 0.989 ± 0.001, sinusoids 0.991 ± 0.002, moving_avg 0.988 ± 0.002.

## Filter controls (`filter_controls.py`, logistic regression, 15 paired test sets, `filter_controls.json`)

Test sets: 3 held-out human flow cells x 5 species-stratified partitions of the bacterial reads (partition 0 = the diagnostics H1 partition). Random banks: 15 Gaussian kernels with the designed lengths, unit-normalised, same ReLU + pooling.

| Features | AUC (mean ± sd) | Accuracy |
| --- | --- | --- |
| stats | 0.9779 ± 0.0015 | 0.9329 ± 0.0057 |
| filters | 0.8916 ± 0.0026 | 0.8116 ± 0.0048 |
| stats+filters | 0.9887 ± 0.0012 | 0.9595 ± 0.0033 |
| stats+edges | 0.9852 ± 0.0007 | 0.9503 ± 0.0050 |
| stats+filters-edges | 0.9866 ± 0.0015 | 0.9520 ± 0.0046 |
| stats+log | 0.9831 ± 0.0016 | 0.9449 ± 0.0063 |
| stats+filters-log | 0.9882 ± 0.0009 | 0.9580 ± 0.0032 |
| stats+sinusoids | 0.9812 ± 0.0015 | 0.9406 ± 0.0068 |
| stats+filters-sinusoids | 0.9885 ± 0.0010 | 0.9605 ± 0.0026 |
| stats+moving_avg | 0.9839 ± 0.0013 | 0.9455 ± 0.0038 |
| stats+filters-moving_avg | 0.9874 ± 0.0010 | 0.9561 ± 0.0039 |
| random0 | 0.9484 ± 0.0020 | 0.8814 ± 0.0040 |
| stats+random0 | 0.9894 ± 0.0005 | 0.9619 ± 0.0025 |
| random1 | 0.9023 ± 0.0044 | 0.8188 ± 0.0084 |
| stats+random1 | 0.9878 ± 0.0006 | 0.9558 ± 0.0036 |
| random2 | 0.9288 ± 0.0033 | 0.8493 ± 0.0042 |
| stats+random2 | 0.9902 ± 0.0005 | 0.9620 ± 0.0026 |
| random3 | 0.9385 ± 0.0015 | 0.8639 ± 0.0056 |
| stats+random3 | 0.9901 ± 0.0006 | 0.9622 ± 0.0031 |
| random4 | 0.9104 ± 0.0029 | 0.8304 ± 0.0062 |
| stats+random4 | 0.9872 ± 0.0012 | 0.9551 ± 0.0048 |

| Paired comparison | Mean AUC difference | Positive / 15 | Paired bootstrap 95% CI, partition 0 (3 folds) |
| --- | --- | --- | --- |
| stats+filters vs stats | +0.0108 ± 0.0013 | 15 | [+0.0103, +0.0141]; [+0.0088, +0.0137]; [+0.0064, +0.0112] |
| stats+filters vs filters | +0.0971 ± 0.0033 | 15 | [+0.0906, +0.1039]; [+0.0926, +0.1075]; [+0.0878, +0.1020] |
| stats+edges vs stats | +0.0073 ± 0.0011 | 15 | [+0.0073, +0.0100]; [+0.0050, +0.0084]; [+0.0051, +0.0083] |
| stats+log vs stats | +0.0053 ± 0.0005 | 15 | [+0.0042, +0.0066]; [+0.0047, +0.0078]; [+0.0028, +0.0057] |
| stats+sinusoids vs stats | +0.0033 ± 0.0004 | 15 | [+0.0023, +0.0047]; [+0.0021, +0.0049]; [+0.0013, +0.0041] |
| stats+moving_avg vs stats | +0.0060 ± 0.0009 | 15 | [+0.0050, +0.0077]; [+0.0046, +0.0083]; [+0.0032, +0.0064] |
| stats+filters vs stats+filters-edges | +0.0021 ± 0.0006 | 15 | [+0.0019, +0.0033]; [+0.0009, +0.0024]; [+0.0010, +0.0028] |
| stats+filters vs stats+filters-log | +0.0005 ± 0.0004 | 14 | [+0.0001, +0.0011]; [+0.0005, +0.0018]; [-0.0011, +0.0010] |
| stats+filters vs stats+filters-sinusoids | +0.0003 ± 0.0003 | 11 | [-0.0006, +0.0005]; [-0.0003, +0.0009]; [-0.0002, +0.0009] |
| stats+filters vs stats+filters-moving_avg | +0.0013 ± 0.0003 | 15 | [+0.0009, +0.0022]; [+0.0007, +0.0020]; [+0.0001, +0.0016] |
| stats+random0 vs stats | +0.0116 ± 0.0015 | 15 | [+0.0108, +0.0150]; [+0.0079, +0.0130]; [+0.0088, +0.0132] |
| stats+random1 vs stats | +0.0099 ± 0.0014 | 15 | [+0.0087, +0.0126]; [+0.0080, +0.0126]; [+0.0069, +0.0113] |
| stats+random2 vs stats | +0.0123 ± 0.0015 | 15 | [+0.0122, +0.0162]; [+0.0087, +0.0141]; [+0.0089, +0.0131] |
| stats+random3 vs stats | +0.0123 ± 0.0014 | 15 | [+0.0107, +0.0151]; [+0.0090, +0.0147]; [+0.0088, +0.0137] |
| stats+random4 vs stats | +0.0093 ± 0.0013 | 15 | [+0.0083, +0.0117]; [+0.0070, +0.0117]; [+0.0064, +0.0103] |
| stats+filters vs stats+random0 | -0.0007 ± 0.0009 | 4 | [-0.0021, +0.0006]; [-0.0007, +0.0024]; [-0.0039, -0.0003] |
| stats+filters vs stats+random1 | +0.0009 ± 0.0009 | 11 | [+0.0002, +0.0030]; [-0.0003, +0.0025]; [-0.0020, +0.0017] |
| stats+filters vs stats+random2 | -0.0015 ± 0.0008 | 0 | [-0.0031, -0.0007]; [-0.0015, +0.0014]; [-0.0039, -0.0005] |
| stats+filters vs stats+random3 | -0.0014 ± 0.0008 | 0 | [-0.0020, +0.0007]; [-0.0022, +0.0012]; [-0.0043, -0.0006] |
| stats+filters vs stats+random4 | +0.0016 ± 0.0005 | 15 | [+0.0010, +0.0033]; [+0.0008, +0.0031]; [-0.0011, +0.0022] |

## Conclusions

1. Feature extraction is necessary: both classifiers are close to chance on the raw window.
2. A linear model is enough: logistic regression on all 280 features matches the MLP (AUC 0.989 vs 0.990; accuracy 0.960 vs 0.951). It is now the default classifier.
3. The convolutional features add a small, consistent gain over the statistics (+0.011 AUC, 15/15 test sets).
4. The filter design does not matter: random kernels of the same lengths add as much (+0.009 to +0.012), and the designed bank sits inside the spread of the random banks (-0.0015 to +0.0016). This matches ROCKET (Dempster et al., 2020): convolution plus pooling at several scales is what helps.
5. Families overlap: dropping any one family costs at most 0.002 AUC; dropping the sinusoids makes no consistent difference.
6. The MLP does better than logistic regression on the statistics alone (0.985 vs 0.978), so the convolutional features give a linear model information that a nonlinear model can partly recover from the statistics.

Caveat: one task, one sample of 24,000 reads; the class label is still confounded with instrument and kit (see the confound diagnostics).
