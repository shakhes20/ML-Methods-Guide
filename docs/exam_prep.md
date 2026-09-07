# Oral Exam Preparation Guide

**Format:** 30 minutes | **Style:** Topic → You choose method → Follow-up questions

---

## The Exam Pattern

```
1. Examiner names a chapter topic
2. Gives you a choice of methods
3. YOU pick the one you know best
4. Follow-up questions ONLY on your chosen method
5. Move to next chapter
```

> **Key insight:** You control which method gets discussed. Always pick your strongest one.

---

## Your Strategy — What to Choose

### Chapter 1: Supervised Learning
**One word:** "Prediction"  
**Choose:** Decision Tree

**Opening statement:**
> "A decision tree partitions the feature space into axis-aligned rectangles.
> At each node we ask a yes/no question about one feature, chosen to maximize
> the information gain — the decrease in impurity. Common impurity measures
> are Gini index and entropy. To prevent overfitting we prune the tree."

### Chapter 2: Clustering
**One word:** "Grouping"  
**Choose:** K-Means

**Opening statement:**
> "K-Means partitions data into K clusters by minimizing the total within-cluster
> variance. The algorithm assigns each point to its nearest centroid, recomputes
> centroids as cluster means, and repeats until convergence. It is guaranteed
> to converge because the objective decreases monotonically."

### Chapter 2: PCA
**One word:** "Compression"

**Opening statement:**
> "PCA finds the directions of maximum variance in the data.
> The first principal component is the direction w maximizing wᵀΣw —
> the variance of projected data. This is the leading eigenvector of
> the covariance matrix. We project onto the top k eigenvectors to
> reduce dimension while preserving most variance."

### Chapter 3: Generative Models
**One word:** "Generation"

**Key distinction:**
> "Discriminative models learn P(Y|X) — given an image, predict a label.
> Generative models learn P(X|Y) — given a label, generate an image."

### Chapter 4: Reinforcement Learning
**One word:** "Reward maximization"

**Three key statements to have ready:**
1. What is RL and the MDP setup
2. The Markov property
3. The Bellman equation

---

## Most Likely Follow-Up Questions

### Decision Trees
| Question | Short Answer |
|----------|-------------|
| How do you grow a tree? | Greedily — pick feature maximizing Gain at each node |
| What is Gini index? | Q = 2p(1-p) — measures label mixture in a node |
| What is overfitting? | Tree too deep → memorizes training data |
| How to fix? | Pruning — reduce number of leaves |
| Bias-variance tradeoff? | Small tree = high bias, large tree = high variance |
| What is the generalization bound? | ε_gen ≤ R_s + √(n·log(d+3)/2m) |

### K-Means
| Question | Short Answer |
|----------|-------------|
| How to choose K? | Elbow method — plot objective vs K |
| Does it always converge? | Yes — objective strictly decreases |
| What is the weakness? | Sensitive to initialization, assumes spherical clusters |
| How to fix initialization? | K-Means++ — choose centroids far apart |
| Difference from hierarchical? | K-Means needs K in advance, hierarchical builds full tree |

### PCA
| Question | Short Answer |
|----------|-------------|
| What does PC1 capture? | Direction of maximum variance |
| How many components to keep? | Enough to explain 90-95% of variance |
| Why use PCA before KNN? | Reduces dimensionality → less curse of dimensionality |

### Reinforcement Learning
| Question | Short Answer |
|----------|-------------|
| What is the Markov property? | Future depends only on current state, not history |
| What is the Bellman equation? | V^π(s) = Σ_a π(a\|s) Σ_{s',r} p(s',r\|s,a)[r + γV^π(s')] |
| What is TD error? | δ = R_{t+1} + γV(S_{t+1}) - V(Sₜ) |
| SARSA vs Q-Learning? | SARSA: on-policy, uses actual next action. Q-Learning: off-policy, uses max |

---

## Key Proofs (if asked)

### Bayes Classifier Optimality
```
1. Write ε_gen(h) = ∫ P(Y≠h(X)|X=x) · μ_X(dx)  (disintegration)
2. For any h: P(Y≠h(X)|X=x) ≥ min{p_x, 1-p_x}  (pointwise lower bound)
3. Bayes classifier achieves = min{p_x, 1-p_x}   (achieves the bound)
4. Integrate → ε_gen(h_μ) ≤ ε_gen(h) for any h  □
```

### Bellman Equation Derivation
```
V^π(s) = E[Σ γᵏ R_{t+k+1} | Sₜ=s]
        = E[R_{t+1} + γ Σ γᵏ R_{t+k+2} | Sₜ=s]
        = E[R_{t+1} + γ V^π(S_{t+1}) | Sₜ=s]
        = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V^π(s')]  □
```

### Bagging Variance Reduction
```
Var[h_agg] = Var[(1/B) Σ h_{s̃_b}]
           = (1/B²) [B·σ² + B(B-1)·Cov]
           = σ²/B + (1-1/B)·ρσ²
           = σ²(ρ + (1-ρ)/B)
As B→∞: → σ²ρ  □
```

---

## The Day Before — Final Checklist

```
✅ Can I state the objective of each chapter in one word?
✅ Can I explain Decision Trees in 5 sentences?
✅ Can I explain K-Means in 3 steps?
✅ Can I write the Bellman equation?
✅ Can I explain the difference between SARSA and Q-Learning?
✅ Can I say what the Markov property is?
✅ Do I know which method to choose in each chapter?
```

---

## Mindset

```
The examiner wants to see:
   ✅ Big picture understanding
   ✅ Explaining ideas in your own words
   ✅ Knowing key formulas and what they mean
   ✅ Honesty when uncertain

He does NOT expect:
   ❌ Perfect memorization of every theorem
   ❌ Full proofs of every result
   ❌ Expert-level depth on all chapters equally

Your advantage:
   You have studied the theory deeply
   You have practical Python experience
   You know the bias-variance tradeoff intuitively
   You can connect theory to practice
```
