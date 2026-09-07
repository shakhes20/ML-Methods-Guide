# Chapter 2: Unsupervised Learning

**One word objective: Grouping / Compression**

No labels — only features `x₁,...,xₘ ∈ Rᵈ`.

Tasks:
- **(a) Clustering** — find group structure
- **(b) Dimensionality Reduction** — find relevant features

> **Important:** Always normalize features first!
> ```
> x̃ᵢₖ = (xᵢₖ - x̄ₖ) / σₖ
> ```

---

## 2.1 Clustering

**Goal:** group similar objects together, dissimilar objects apart.

### Distance Measures

| Name | Formula |
|------|---------|
| Euclidean | `‖xᵢ - xⱼ‖` |
| Manhattan | `Σₖ |xᵢₖ - xⱼₖ|` |
| Maximum | `maxₖ |xᵢₖ - xⱼₖ|` |
| Minkowski | `(Σₖ |xᵢₖ - xⱼₖ|^q)^(1/q)` |

---

### 2.1.1 Hierarchical Agglomerative Clustering

**Core idea:** bottom-up merging — start with each point as its own cluster.

**Algorithm:**
```
Initialize: C = {{x₁}, {x₂}, ..., {xₘ}}

For ℓ = 1,...,m-1:
   1. Find two closest clusters Cᵢ, Cⱼ
   2. Merge: Cᵢ ∪ Cⱼ → new cluster
   3. Update distance matrix

Result: dendrogram (tree of merges)
```

**Linkage methods** — how to measure distance between clusters:

| Linkage | Distance(Cᵢ, Cⱼ) | Behavior |
|---------|------------------|---------|
| Single | min distance between any two points | Chaining effect |
| Complete | max distance between any two points | Compact clusters |
| Average | mean of all pairwise distances | Balanced |
| Ward | increase in within-cluster variance | Best for globular clusters |

**How to choose number of clusters:**  
Cut the dendrogram at the level with the largest gap between merges.

---

### 2.1.2 K-Means Clustering

**Core idea:** partition into K clusters minimizing within-cluster variance.

**Objective:**
```
min_{C₁,...,Cₖ}  Σⱼ Σ_{xᵢ ∈ Cⱼ} ‖xᵢ - μⱼ‖²

where μⱼ = (1/|Cⱼ|) Σ_{xᵢ ∈ Cⱼ} xᵢ  (centroid)
```

**Algorithm:**
```
Initialize: choose K centroids μ₁,...,μₖ randomly

Repeat:
   ASSIGN: assign each xᵢ to nearest centroid
           Cⱼ = {xᵢ : j = argmin_k ‖xᵢ - μₖ‖}

   UPDATE: recompute centroids
           μⱼ = (1/|Cⱼ|) Σ_{xᵢ ∈ Cⱼ} xᵢ

Until: assignments stop changing
```

**Convergence:** guaranteed — objective decreases monotonically, finite states.

**How to choose K:**
- **Elbow method:** plot objective vs K, pick the "elbow"
- **Silhouette score:** measure how well each point fits its cluster
- **Gap statistic:** compare to random baseline

**Weaknesses:**
- Sensitive to initialization → use **K-Means++**
- Assumes spherical, equal-size clusters
- Must specify K in advance
- Can get stuck in local optima

**K-Means++ initialization:**
```
1. Choose first centroid uniformly at random
2. Choose next centroid with probability ∝ distance² to nearest existing centroid
3. Repeat until K centroids chosen
```

---

### 2.1.3 Spectral Clustering

**Core idea:** cluster in the space of graph eigenvectors — works for non-convex shapes.

**Why:** K-Means fails for clusters like crescents or rings.  
Spectral finds clusters based on **connectivity**, not distance.

**Algorithm sketch:**
```
1. Build similarity graph W (e.g., Gaussian kernel)
2. Compute graph Laplacian L = D - W
3. Compute eigenvectors of L
4. Run K-Means on the eigenvector representation
```

**When to use:** when clusters are non-convex or have complex shapes.

---

### 2.1.4 Comparison of Clustering Methods

| Method | Shape | Scalable | Need K? | Deterministic |
|--------|-------|---------|---------|---------------|
| Hierarchical | Any | No (O(m²)) | No | Yes |
| K-Means | Convex | Yes | Yes | No |
| Spectral | Any | No | Yes | Yes |

---

## 2.2 Dimensionality Reduction

**Goal:** represent data in fewer dimensions while preserving structure.

---

### 2.2.1 Principal Component Analysis (PCA)

**Core idea:** find directions of maximum variance.

**Objective:**
```
max_{w: ‖w‖=1}  wᵀΣw

where Σ = (1/(m-1)) Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ  (covariance matrix)
```

**Solution:** w₁ = leading eigenvector of Σ (largest eigenvalue).

**Algorithm:**
```
1. Center data: x̃ᵢ = xᵢ - x̄
2. Compute covariance matrix Σ
3. Compute eigenvectors v₁,...,vₐ of Σ (sorted by eigenvalue)
4. Project: zᵢ = [v₁ᵀxᵢ, v₂ᵀxᵢ,...,vₖᵀxᵢ]  (k < d)
```

**How many components to keep:**
```
Explained variance ratio = λₖ / Σⱼ λⱼ

Keep enough components to explain 90-95% of total variance
```

**Geometric interpretation:**
```
PC1: direction of maximum spread in data
PC2: direction of maximum spread, orthogonal to PC1
...
```

**Why useful:**
- Remove redundant/correlated features
- Visualization (project to 2D or 3D)
- Speed up downstream algorithms (e.g., KNN in lower dimension)
- Noise reduction

**Connection to KNN:**  
Apply PCA first → fewer dimensions → less curse of dimensionality → KNN works better.

---

### 2.2.2 Compressed Sensing

**Core idea:** recover a sparse signal from few measurements.

**Key assumption:** signal x is **sparse** — most entries are zero.

**Recovery:** solve LASSO:
```
min ‖x‖₁  subject to  Ax = b
```

**Key condition:** measurement matrix A must satisfy **Restricted Isometry Property (RIP)**.

**Application:** MRI imaging, signal compression, recommendation systems.

---

## Summary

```
CLUSTERING
   Hierarchical → bottom-up merging, dendrogram, no K needed
   K-Means      → minimize within-cluster variance, need K, fast
   Spectral     → graph eigenvectors, non-convex shapes

DIMENSIONALITY REDUCTION
   PCA          → maximum variance directions, eigenvectors of Σ
   Compressed Sensing → sparse recovery, LASSO

KEY CONNECTIONS
   Normalize first → always
   PCA before KNN → reduces curse of dimensionality
   K-Means after PCA → often better results
```
