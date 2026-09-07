# Formula Sheet — Methods in Machine Learning

---

## Chapter 1: Supervised Learning

### Setup
```
ε_gen(h) = R_μ(h) = E_μ[ℓ(h,(X,Y))]   ← true error (unknown)
R_s(h) = (1/m) Σᵢ ℓ(h,(xᵢ,yᵢ))         ← training error (computable)
Gap = ε_gen - R_s                         ← generalization gap
```

### Bayes Classifier
```
h_μ(x) = +1  if  p(x) ≥ 1/2,  else -1
Local error = min{p_x, 1-p_x}
Bayes error = R_μ(h_μ) = irreducible floor
```

### KNN
```
h_s^(k)(x) = sgn(Σᵢ₌₁ᵏ y_{π_x(i)})
Bound: E[ε_gen] ≤ 2R_μ(h_μ) + 2L√d · m^{-1/(d+1)}
Data needed: m ~ (√d/ε)^{d+1}
```

### Decision Trees
```
Gini:    Q = 2p(1-p)
Entropy: Q = -p·log(p) - (1-p)·log(1-p)
Gain:    Q(parent) - [|s₀|/|s|·Q(s₀) + |s₁|/|s|·Q(s₁)]
MDI_k:   Σ_nodes (|sⱼ|/|s|)·Gain(sⱼ,k)
Bound:   ε_gen ≤ R_s + √[(n·log(d+3) + log(2/δ))/2m]
Pruning: argmin_h Σⱼ (|sⱼ|/|s|)Q(sⱼ) + α·n
```

### AdaBoost
```
ε_t = Σ_{wrong} wᵢ^(t)
α_t = ½ log((1-ε_t)/ε_t)
wᵢ^(t+1) ∝ wᵢ^(t) · exp(-α_t · yᵢ · h_t(xᵢ))
Output: h_s(x) = sgn(Σ_t α_t h_t(x))
Bound: R_s ≤ exp(-2γ²T)   where γ = min_t(½ - ε_t)
VCD(H_{T,B}) ∈ O(T·B·log(TB))
```

### Bagging / Random Forest
```
Var[h_agg] = σ²·(ρ + (1-ρ)/B)
As B→∞: Var → σ²ρ
p = ⌊√d⌋ (classification),  p = ⌊d/3⌋ (regression)
OOB error → LOO-CV error as B→∞
```

---

## Chapter 2: Unsupervised Learning

### Normalization
```
x̃ᵢₖ = (xᵢₖ - x̄ₖ) / σₖ
```

### K-Means Objective
```
min_{C₁,...,Cₖ} Σⱼ Σ_{xᵢ∈Cⱼ} ‖xᵢ - μⱼ‖²
μⱼ = (1/|Cⱼ|) Σ_{xᵢ∈Cⱼ} xᵢ  (centroid update)
```

### PCA
```
max_{‖w‖=1} wᵀΣw → w₁ = leading eigenvector of Σ
Σ = (1/(m-1)) Σᵢ (xᵢ-x̄)(xᵢ-x̄)ᵀ
Explained variance ratio = λₖ / Σⱼ λⱼ
```

---

## Chapter 3: Generative Models

### Key Distinction
```
Discriminative: P(Y|X)    → prediction
Generative:     P(X|Y)    → generation
Generator: G(y,Z) ~ P(X|Y=y),  Z ~ N(0,Iₙ)
```

### GMM
```
P(X|Y=y) = N(μᵧ, Σᵧ)
μ̂ᵧ = (1/mᵧ) Σ_{i:yᵢ=y} xᵢ
Σ̂ᵧ = (1/(mᵧ-1)) Σ (xᵢ-μ̂ᵧ)(xᵢ-μ̂ᵧ)ᵀ
ŵᵧ = mᵧ/m
Generate: X = μ̂ᵧ + Σ̂ᵧ^(1/2) · Z,  Z~N(0,I)
```

### GAN
```
min_G max_D  E[log D(x)] + E[log(1-D(G(z)))]
D: real vs fake classifier
G: fools D by generating realistic data
```

---

## Chapter 4: Reinforcement Learning

### MDP
```
(S, A, R, p, γ)
p(s',r|s,a) = P(S_{t+1}=s', R_{t+1}=r | Sₜ=s, Aₜ=a)
Markov property: future ⊥ past | current state
```

### Value Functions
```
V^π(s) = E_π[Σ_{k=0}^∞ γᵏ R_{t+k+1} | Sₜ=s]
Q^π(s,a) = E_π[Σ_{k=0}^∞ γᵏ R_{t+k+1} | Sₜ=s, Aₜ=a]
```

### Bellman Equations
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γV^π(s')]
V*(s)  = max_a Σ_{s',r} p(s',r|s,a) [r + γV*(s')]
π*(s)  = argmax_a Σ_{s',r} p(s',r|s,a) [r + γV*(s')]
```

### TD Learning
```
δₜ = R_{t+1} + γV(S_{t+1}) - V(Sₜ)          ← TD error
V(Sₜ) ← V(Sₜ) + α·δₜ                         ← TD(0) update

SARSA:      Q(Sₜ,Aₜ) ← Q + α[R_{t+1} + γQ(S_{t+1},A_{t+1}) - Q]
Q-Learning: Q(Sₜ,Aₜ) ← Q + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q]
```

---

## Bias-Variance Decomposition

```
E[(Y - h_S(x))²] = σ² + Bias²(x) + Variance(x)

σ²        = E[(Y - h_μ(x))²]                    ← irreducible noise
Bias²     = (h_μ(x) - E_S[h_S(x)])²             ← systematic error
Variance  = E_S[(h_S(x) - E_S[h_S(x)])²]        ← instability

Complex model: Bias↓ Variance↑
Simple model:  Bias↑ Variance↓
```

---

## VC Dimension

```
VCD(H) = max set size that H can shatter
H learnable ⟺ VCD(H) < ∞

ε_gen ≤ R_s + O(√(VCD/m))

Common VCDs:
   Threshold in R:        1
   Linear in Rᵈ:         d+1
   Decision tree n leaves: O(n log n)
   KNN:                   ∞
   AdaBoost T rounds:     O(T·B·log(TB))
```
