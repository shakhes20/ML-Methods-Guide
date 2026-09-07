# Chapter 1: Supervised Learning

**One word objective: Prediction**

---

## Setup

| Symbol | Meaning |
|--------|---------|
| `X` | Feature space |
| `Y` | Label space |
| `μ` | Unknown data distribution |
| `s` | Training sample `{(x₁,y₁),...,(xₘ,yₘ)}` |
| `h: X → Y` | Hypothesis (the model) |
| `ε_gen(h)` | Generalization error = true error on all data |
| `R_s(h)` | Empirical risk = training error |

**Goal:** learn `h` minimizing `ε_gen(h) = E_μ[ℓ(h,(X,Y))]`

---

## 0. Bayes Classifier — The Theoretical Benchmark

**Definition:**
```
h_μ(x) = +1  if  P_μ(Y=+1 | X=x) ≥ 1/2
          -1  otherwise
```

**Why optimal:**  
By disintegration of μ, the risk decomposes pointwise:
```
R_μ(h) = ∫ P_μ(Y ≠ h(X) | X=x) · μ_X(dx)
```
The Bayes classifier minimizes the integrand at EVERY x → minimizes the whole integral.

**Local error of Bayes classifier:**
```
P_μ(Y ≠ h_μ(X) | X=x) = min{p_x, 1 - p_x}
```

**Bayes error** = irreducible floor. No algorithm can beat it.

---

## 1.1 k-Nearest Neighbor (KNN)

**Core idea:** "Things that look alike must be alike."  
Learning by memorization — no training, just store data.

**Requires:** a metric space (X, ρ), e.g., Euclidean distance.

**Permutation:** rank training points by distance from query x:
```
ρ(x, x_{π_x(1)}) ≤ ρ(x, x_{π_x(2)}) ≤ ... ≤ ρ(x, x_{π_x(m)})
```

**Classifier:**
```
h_s^(k)(x) = +1  if  Σᵢ₌₁ᵏ y_{π_x(i)} ≥ 0
              -1  otherwise
```

**Regression version:**
```
h_s^(k)(x) = (1/k) Σᵢ₌₁ᵏ y_{π_x(i)}
```

### Key Theorem 1.7 — Generalization Bound
Assuming p(x) is L-Lipschitz:
```
E[ε_gen(h_S^(1))] ≤ 2·R_μ(h_μ) + 2L√d · m^{-1/(d+1)}
```

**Curse of Dimensionality:**  
To get error within ε of Bayes, need:
```
m ~ (√d / ε)^(d+1)   ← exponential in dimension d!
```

### Pros and Cons
| ✅ Pros | ❌ Cons |
|--------|--------|
| Simple, no training | Slow prediction O(md) |
| Works for any distribution | Fails in high dimensions |
| Nonparametric | No feature importance |

---

## 1.2 Decision Trees

**Core idea:** partition feature space into axis-aligned rectangles, predict constant in each.

**Formal definition:**
```
h(x) = Σⱼ cⱼ · 1_{Rⱼ}(x)
```
where Rⱼ are disjoint rectangles covering X.

**Leaf predictions:**
- Classification: majority vote in leaf
- Regression: mean of labels in leaf

### Growing a Tree — Impurity Gain

**Impurity measures** (for fraction p of positive labels):

| Measure | Formula | Used by |
|---------|---------|---------|
| Gini | `Q = 2p(1-p)` | CART |
| Entropy | `Q = -p·log(p) - (1-p)·log(1-p)` | ID3, C4.5 |
| Misclassification | `Q = 1 - max(p, 1-p)` | — |

**Information Gain:**
```
Gain(s, k) = Q(s) - [ |s₀|/|s| · Q(s₀) + |s₁|/|s| · Q(s₁) ]
```
Pick feature k* = argmax Gain(s, k).

### Generalization Bound (Theorem 1.12)
```
P( ε_gen(h) ≤ R_s(h) + √[(n·log(d+3) + log(2/δ)) / 2m] ) ≥ 1-δ
```

**Bias-variance tradeoff:**
- Large n → low R_s(h) but large penalty → overfitting
- Small n → high R_s(h) but small penalty → underfitting

### Pruning
**Cost-complexity pruning:**
```
h_α = argmin_{h ⊂ h_full}  Σⱼ (|sⱼ|/|s|)·Q(sⱼ) + α·n_h
```
α controls regularization strength — choose by cross-validation.

### Feature Importance — MDI
```
MDI_k = Σ_{nodes j using k} (|sⱼ|/|s|) · Gain(sⱼ, k)
```

### Pros and Cons
| ✅ Pros | ❌ Cons |
|--------|--------|
| Interpretable | High variance |
| Handles mixed inputs | Lower accuracy alone |
| Built-in feature importance | NP-hard to find optimal tree |
| Robust to outliers | Axis-aligned boundaries only |

---

## 1.3 Boosting — AdaBoost

**Core idea:** combine many weak learners sequentially, focusing on mistakes.

**Weak learner:** just better than random → R_s(h) ≤ 1/2 - γ

**Name:** ADAptive Boosting (Schapire & Freund, 1995)

### Algorithm
```
Initialize: w_i^(1) = 1/m  for all i

For t = 1,...,T:
   1. Train: h_t = A(s, w^(t))
   2. Error: ε_t = Σ_{wrong} w_i^(t)
   3. Weight: α_t = ½ log((1-ε_t) / ε_t)
   4. Update: w_i^(t+1) ∝ w_i^(t) · exp(-α_t · y_i · h_t(x_i))
              normalize so Σ w_i = 1

Output: h_s(x) = sgn( Σₜ αₜ hₜ(x) )
```

### Understanding α_t

| ε_t | Meaning | α_t |
|-----|---------|-----|
| 0.5 | Random guessing | 0 (ignored) |
| 0.3 | Decent learner | 0.42 |
| 0.1 | Good learner | 1.10 |
| 0.01 | Near perfect | 2.30 |

### Training Error Bound (Theorem 1.13)
```
R_s(h_s) ≤ exp(-2γ²T)
```
Training error decreases **exponentially** with rounds T.

### Key Facts
- AdaBoost = gradient descent on **exponential loss**
- Sensitive to **outliers** (exponential upweighting)
- T controls bias-variance tradeoff
- Predecessor to XGBoost, LightGBM, CatBoost

---

## 1.4 Bagging and Random Forests

**Core idea:** reduce variance by averaging many trees.

### Bias-Variance Decomposition
```
E[|h_μ(x) - h_S(x)|²] = Var[h_S(x)] + Bias²(x) + σ²
```
- Bagging reduces **variance** — does NOT change bias

### Bagging Algorithm
```
For b = 1,...,B:
   1. Draw bootstrap sample s̃_b from s
   2. Train full tree h_{s̃_b} on s̃_b

Predict:
   Classification: majority vote
   Regression: average
```

### Variance Reduction (Proposition 1.15)
```
Var[h_agg] = σ²·(ρ + (1-ρ)/B)

As B → ∞:  Var → σ²ρ
```
Lower correlation ρ between trees → better variance reduction.

### Random Forest — Decorrelating Trees
At each node: randomly select **p ≤ d** features, split on best among those.

**Standard choices:**
```
p = ⌊√d⌋   for classification
p = ⌊d/3⌋  for regression
```

Effect of p:
- Smaller p → less correlated trees → lower variance
- Smaller p → worse splits → higher bias
- Sweet spot found by cross-validation or OOB error

### Out-of-Bag (OOB) Error
```
For each (xᵢ, yᵢ): predict using only trees where (xᵢ,yᵢ) ∉ s̃_b

As B → ∞: OOB error → LOO-CV error
```
Free internal estimate of generalization error!

---

## Comparison: All Chapter 1 Methods

| Property | KNN | Decision Tree | AdaBoost | Random Forest |
|----------|-----|--------------|---------|---------------|
| Core idea | Neighbors | Questions | Combine weak | Average trees |
| Bias | Low | Tunable | Decreases with T | Moderate |
| Variance | High in high-d | High if large | Tunable | Low |
| Interpretable | No | Yes | No | No |
| Key parameter | k | n leaves | T rounds | B, p |
| Solves | — | Bias+Variance | High bias | High variance |
