# Bias-Variance Tradeoff — Deep Dive

---

## The Decomposition

For squared loss, the expected error decomposes as:

```
E[(Y - h_S(x))²] = σ² + Bias²(x) + Variance(x)
```

| Term | Formula | Meaning | Can reduce? |
|------|---------|---------|------------|
| σ² | E[(Y - h_μ(x))²] | Irreducible noise | ❌ Never |
| Bias² | (h_μ(x) - E_S[h_S(x)])² | Systematic error from model being too simple | ✅ More complex model |
| Variance | E_S[(h_S(x) - E_S[h_S(x)])²] | Instability across training sets | ✅ Simpler model or ensemble |

---

## The Tradeoff

```
SIMPLE MODEL (e.g., decision stump, k=large in KNN):
   Bias:     HIGH  — model too rigid to capture true pattern
   Variance: LOW   — always gives similar answer regardless of training data
   Gap:      small — train ≈ test
   Both:     high  → underfitting

COMPLEX MODEL (e.g., deep tree, k=1 in KNN):
   Bias:     LOW   — flexible enough to approximate true pattern
   Variance: HIGH  — sensitive to which training samples you got
   Gap:      large — train << test
   Train:    low   → Test: high → overfitting

SWEET SPOT:
   Bias and Variance balanced
   Training error moderate
   Test error minimized
```

---

## How to Diagnose

| Observation | Diagnosis | Fix |
|-------------|-----------|-----|
| Train HIGH, Test HIGH | High bias | More complex model |
| Train LOW, Test HIGH | High variance | Simpler model, more data, regularize |
| Train LOW, Test LOW | ✅ Good balance | Ship it |
| Large Train-Test gap | Variance | Pruning, bagging, regularization |

---

## In Each Method

| Method | Bias controlled by | Variance controlled by |
|--------|-------------------|----------------------|
| KNN | k (larger k → more bias) | k (smaller k → more variance) |
| Decision Tree | n leaves (fewer → more bias) | n leaves (more → more variance) |
| AdaBoost | T rounds (fewer → more bias) | T rounds (more → more variance) |
| Random Forest | p features, tree depth | B trees, p features |

---

## What You Can and Cannot Observe

```
OBSERVABLE:
   R_s(h)    = training error   → compute directly
   R_test(h) = test error       → compute on held-out set
   Gap       = test - train     → indicates variance problem

NOT OBSERVABLE:
   Bias²     = needs h_μ (optimal hypothesis) — unknown
   σ²        = needs true distribution μ — unknown
   ε_gen     = estimated by R_test, bounded by generalization bound
```

---

## Connection to Generalization Bound

```
ε_gen(h) ≤ R_s(h) + √[(n·log(d+3) + log(2/δ))/2m]
           ↑                  ↑
        captures           captures
        bias term          variance term

Decreasing n (pruning):
   R_s(h) goes UP   ← more bias
   Penalty goes DOWN ← less variance
   → find optimal n that minimizes the sum
```
