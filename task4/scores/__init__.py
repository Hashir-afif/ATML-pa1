"""Post-hoc novelty scores. Every score is an **unknownness** u(x): larger = more novel (manual section 4, step 2).

All four read the *same* saved logits/features of the frozen model, so they differ only in the statistic used.

* MSP      u = 1 - max_k softmax(z)_k          (normalised confidence)
* MLS      u = -max_k z_k                      (absolute logit magnitude)
* Energy   u = -log sum_k exp(z_k)             (aggregate logit evidence)
* Mahalanobis u = min_c (f - mu_c)^T Sigma^-1 (f - mu_c), with class means and one shared **diagonal** covariance
  estimated from unaugmented CIFAR-10 **training** features, plus 1e-6 on every diagonal entry.
"""
from task4.scores.energy import energy
from task4.scores.mahalanobis import MahalanobisScore
from task4.scores.mls import mls
from task4.scores.msp import msp

POST_HOC = {"msp": msp, "mls": mls, "energy": energy}
