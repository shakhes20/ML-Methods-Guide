# 🤖 Methods in Machine Learning — Study Guide

> A structured reference for the **Methods in Machine Learning** course  
> TU Bergakademie Freiberg | Prof. Dr. Björn Sprungk | Summer Term 2026

---

## 📚 Course Structure

| Chapter | Topic | Key Methods |
|---------|-------|-------------|
| [Chapter 1](chapters/01_supervised_learning.md) | Supervised Learning | KNN, Decision Trees, Boosting, Bagging |
| [Chapter 2](chapters/02_unsupervised_learning.md) | Unsupervised Learning | Clustering, PCA, Compressed Sensing |
| [Chapter 3](chapters/03_generative_models.md) | Generative Models | GMM, GAN, VAE |
| [Chapter 4](chapters/04_reinforcement_learning.md) | Reinforcement Learning | MDP, DP, TD Learning, Q-Learning |

---

## 🎯 Quick Reference — One Word per Topic

| Topic | Objective |
|-------|-----------|
| Supervised Learning | **Prediction** |
| Clustering | **Grouping** |
| Dimensionality Reduction | **Compression** |
| Generative Models | **Generation** |
| Reinforcement Learning | **Reward Maximization** |

---

## 🧮 Key Formulas at a Glance

### Bayes Classifier
```
h_μ(x) = +1  if  P(Y=+1 | X=x) ≥ 1/2
Local error  = min{p_x, 1 - p_x}
```

### Generalization Bound (Decision Trees)
```
ε_gen(h) ≤ R_s(h) + √[ (n·log(d+3) + log(2/δ)) / 2m ]
```

### AdaBoost
```
ε_t  = Σ_{wrong} w_i          (weighted error)
α_t  = ½ log((1-ε_t) / ε_t)  (trust score)
w_i ↑ if misclassified, w_i ↓ if correct
Training bound: R_s ≤ exp(-2γ²T)
```

### Bagging Variance Reduction
```
Var[h_agg] = σ²(ρ + (1-ρ)/B)
As B → ∞:  Var → σ²ρ
```

### Bellman Equation
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V^π(s')]
```

### TD Error
```
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
V(S_t) ← V(S_t) + α · δ_t
```

---

## 📖 Textbooks

| Book | Authors | Free Link |
|------|---------|-----------|
| Understanding Machine Learning | Shalev-Shwartz & Ben-David | [PDF](https://www.cse.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| Elements of Statistical Learning | Hastie, Tibshirani, Friedman | [PDF](https://web.stanford.edu/~hastie/ElemStatLearn/) |
| Reinforcement Learning: An Introduction | Sutton & Barto | [PDF](http://incompleteideas.net/book/the-book.html) |
| Foundations of Machine Learning | Mohri et al. | [PDF](https://cs.nyu.edu/~mohri/mlbook/) |
| Deep Learning | Goodfellow et al. | [Web](https://www.deeplearningbook.org/) |

---

## 🎥 Video Resources

| Topic | Channel | Search Term |
|-------|---------|-------------|
| Decision Trees | StatQuest | `StatQuest Decision Trees Clearly Explained` |
| AdaBoost | StatQuest | `StatQuest AdaBoost Clearly Explained` |
| Random Forests | StatQuest | `StatQuest Random Forests Part 1` |
| K-Means | StatQuest | `StatQuest K-means Clustering` |
| PCA | StatQuest | `StatQuest PCA Step by Step` |
| RL | David Silver | `David Silver RL Lecture UCL` |

---

## 🗂️ Repository Structure

```
ML-Methods-Guide/
│
├── README.md                         ← You are here
│
├── chapters/
│   ├── 01_supervised_learning.md     ← KNN, Trees, Boosting, Bagging
│   ├── 02_unsupervised_learning.md   ← Clustering, PCA
│   ├── 03_generative_models.md       ← GMM, GAN
│   └── 04_reinforcement_learning.md  ← MDP, DP, TD, Q-Learning
│
├── docs/
│   ├── bias_variance.md              ← Bias-variance deep dive
│   ├── generalization_bounds.md      ← All bounds in one place
│   ├── vc_dimension.md               ← VC theory
│   └── exam_prep.md                  ← Oral exam strategy
│
├── notebooks/
│   ├── decision_tree_example.ipynb   ← Student pass/fail dataset
│   └── random_forest_example.ipynb   ← Hyperparameter comparison
│
└── assets/
    └── formula_sheet.md              ← One-page formula reference
```

---

## ✍️ How to Use This Guide

1. **Before class** — read the relevant chapter file
2. **During study** — use the formula sheet for quick reference
3. **Before exam** — read `docs/exam_prep.md`
4. **For practice** — run the Jupyter notebooks

---

*Built during MeML course preparation, Summer 2026*
