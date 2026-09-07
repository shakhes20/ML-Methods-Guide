# Chapter 3: Generative Models

**One word objective: Generation**

---

## Discriminative vs Generative — The Key Distinction

| | Discriminative | Generative |
|--|---------------|-----------|
| **Learns** | P(Y \| X) | P(X \| Y) |
| **Given** | Image → predict label | Label → generate image |
| **Question** | "Is this a cat?" | "Generate a cat image" |
| **Examples** | KNN, Trees, SVM, Neural nets | GMM, GAN, VAE, Diffusion |

**Generative model formally:**  
Learn generator `G: Y × Z → X` such that:
```
G(y, Z) ~ P(X | Y=y)   where Z ~ P_Z (latent variable)
```
Latent space Z is typically `Z ~ N(0, Iₙ)`.

---

## 3.1 Gaussian Mixture Models (GMM)

**Core idea:** model each class as a Gaussian distribution.

**Assumption:**
```
P(X | Y=y) = N(μᵧ, Σᵧ)
```

**Full distribution of X:**
```
X ~ Σᵧ wᵧ · N(μᵧ, Σᵧ)   where wᵧ = P(Y=y)
```

### Parameter Estimation (MLE)

Given data `(xᵢ, yᵢ)`:

```
μ̂ᵧ  = (1/mᵧ) Σ_{i: yᵢ=y} xᵢ                          ← sample mean

Σ̂ᵧ  = (1/(mᵧ-1)) Σ_{i: yᵢ=y} (xᵢ - μ̂ᵧ)(xᵢ - μ̂ᵧ)ᵀ   ← sample covariance

ŵᵧ  = mᵧ / m                                            ← class proportion
```

### Generating New Samples
```
X = μ̂ᵧ + Σ̂ᵧ^(1/2) · Z,   Z ~ N(0, Iₐ)
```

### Pros and Cons
| ✅ Pros | ❌ Cons |
|--------|--------|
| Simple, closed-form estimation | Assumes Gaussian — may not fit |
| Fast generation | Limited to elliptical clusters |
| Interpretable parameters | Fails for complex distributions |

---

## 3.2 Generative Adversarial Networks (GAN)

**Core idea:** two neural networks compete — Generator vs Discriminator.

**Invented by:** Ian Goodfellow et al., 2014

### The Adversarial Game

```
Generator G:     takes noise z ~ P_Z → produces fake data G(z)
                 Goal: fool the Discriminator

Discriminator D: takes real or fake data → outputs probability of being real
                 Goal: distinguish real from fake
```

**Minimax objective:**
```
min_G max_D  E_{x~P_data}[log D(x)] + E_{z~P_Z}[log(1 - D(G(z)))]
```

**Interpretation:**
- `D(x)` = probability that x is real
- `D(G(z))` = probability that fake image is real
- D wants to maximize (correctly classify)
- G wants to minimize (fool D)

### Training Procedure
```
Repeat:
   1. Sample real data batch from P_data
   2. Sample noise batch z ~ P_Z, generate fake data G(z)
   3. Update D: maximize log D(x) + log(1-D(G(z)))
   4. Update G: minimize log(1-D(G(z)))
             = maximize log D(G(z))  (practical version)
Until: G produces realistic data
```

### Problems with GANs

| Problem | Description | Solution |
|---------|-------------|---------|
| Mode collapse | G generates only few types | Minibatch discrimination |
| Training instability | D wins too fast → no gradient for G | Gradient penalty (WGAN) |
| Non-convergence | Oscillation, no equilibrium | Careful hypertuning |

### GAN Variants

| Model | Key Idea |
|-------|---------|
| DCGAN | Convolutional layers for images |
| WGAN | Wasserstein distance — more stable |
| StyleGAN | Controls style at each layer |
| Conditional GAN | Condition on label y |
| Pix2Pix | Image-to-image translation |

---

## 3.3 Other Deep Generative Models

### Variational Autoencoder (VAE)
```
Encoder: x → z (compress to latent space)
Decoder: z → x̂ (reconstruct)
Loss: reconstruction error + KL divergence
```
Learns a smooth, structured latent space — good for interpolation.

### Normalizing Flows
```
Transform simple distribution (e.g., Gaussian) through
series of invertible transformations to complex distribution.
Exact likelihood computation.
```

### Diffusion Models (e.g., DALL-E, Stable Diffusion)
```
Forward: gradually add Gaussian noise to data
Reverse: learn to denoise step by step
Generation: start from pure noise → denoise → realistic image
```
Currently state-of-the-art for image generation.

---

## Summary

```
DISCRIMINATIVE: P(Y|X) → prediction given input
GENERATIVE:     P(X|Y) → generate input given label

GMM:    simple Gaussian assumption, closed-form estimation
GAN:    adversarial game Generator vs Discriminator
        minimax: min_G max_D objective
VAE:    encoder-decoder with structured latent space
Diffusion: add noise → learn to denoise → generate

KEY CONNECTION TO CHAPTER 1:
   Discriminative models = everything in Chapter 1
   Generative models = model the data itself, not just the boundary
```
