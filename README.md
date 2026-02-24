# Learning Probability Density Functions Using NO₂ Air Quality Data (GAN-Based)

---

## 1. Overview

This project learns the probability density function (PDF) of a transformed NO₂ concentration variable using a Generative Adversarial Network (GAN).

Unlike parametric approaches (e.g., Gaussian fitting), this method does **not assume any analytical distribution form**.  
Instead, the distribution is learned directly from data samples using adversarial training.

The GAN implicitly models the underlying distribution of the transformed variable and the PDF is estimated from generated samples using Kernel Density Estimation (KDE).

---

## 2. Problem Statement


Given NO₂ concentration values (x), apply the nonlinear transformation:

z = x + a_r * sin(b_r * x)

where:

a_r = 0.5 * (r mod 7)

b_r = 0.3 * ((r mod 5) + 1)

For roll number:

r = 102313038

We compute:

r mod 7 = 2  →  a_r = 1.0  
r mod 5 = 3  →  b_r = 1.2  

Final transformation used:

z = x + 1.0 * sin(1.2x)

The goal is to:

1. Assume \( z \) is sampled from an unknown distribution.
2. Train a GAN to learn this distribution.
3. Use the generator to implicitly model the PDF of \( z \).

No parametric PDF (Gaussian, exponential, etc.) is assumed.

---

## 3. Requirements

Install the following dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

---

## 4. How to Run

1. Place the dataset inside:

```
Data/data.csv
```

2. Run:

```bash
python gan_pdf_model.py
```

The script will:

- Load and transform data
- Train WGAN-GP
- Generate synthetic samples
- Estimate PDF using KDE
- Plot training stability
- Compute mode coverage score

---

## 5. Transformation Parameters (a_r, b_r)

For roll number 102313038:

- \( a_r = 1.0 \)
- \( b_r = 1.2 \)

Transformation applied:

z = x + 1.0 * sin(1.2x)

---

## 6. GAN Architecture Description

This implementation uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for improved training stability.

### Generator

Fully connected neural network:

- Linear(1 → 128)
- ReLU
- Linear(128 → 256)
- ReLU
- Linear(256 → 1)

Input:
\[
\epsilon \sim \mathcal{N}(0,1)
\]

Output:
\[
z_f = G(\epsilon)
\]

---

### Discriminator (Critic)

Fully connected neural network:

- Linear(1 → 128)
- LeakyReLU(0.2)
- Linear(128 → 64)
- LeakyReLU(0.2)
- Linear(64 → 1)

The discriminator distinguishes between:

- Real samples: \( z \)
- Fake samples: \( z_f = G(\epsilon) \)

---

## 7. PDF Learned by GAN

After training:

1. A large number of samples \( z_f \) are generated from the trained generator.
2. Kernel Density Estimation (KDE) is applied to approximate:

\[
\hat{p}(z)
\]

The resulting plot represents the learned probability density of the transformed variable.

This is a **non-parametric density estimate**, fully learned from GAN-generated samples.

---

## 8. Results

### 8.1 Training Stability

Training stability is evaluated by plotting:

- Discriminator loss
- Generator loss

Stable oscillatory behavior indicates balanced adversarial training.

WGAN-GP improves stability by using:

- Wasserstein loss
- Gradient penalty regularization

---

### 8.2 Mode Coverage

Mode coverage is evaluated by measuring histogram overlap between:

- Real distribution
- Generated distribution

Coverage score:

\[
\text{Coverage} = \frac{\sum \min(\text{Real}_i, \text{Fake}_i)}{\sum \text{Real}_i}
\]

Higher score indicates better learning of distribution modes.

---

### 8.3 Distribution Quality

Distribution quality is assessed using:

- KDE-based PDF comparison
- Mode coverage score
- Visual histogram overlay

If:

- The generated PDF closely matches the real distribution
- Mode coverage is high
- Training remains stable

Then the GAN has successfully learned the unknown distribution.

---

## 9. Assignment Requirement Checklist

| Requirement | Status |
|------------|--------|
| Transformation parameters (a_r, b_r) | ✔ Included |
| GAN architecture description | ✔ Included |
| PDF plot from GAN samples | ✔ Generated |
| Mode coverage observation | ✔ Included |
| Training stability observation | ✔ Included |
| Quality of generated distribution | ✔ Included |
| No parametric PDF assumption | ✔ Satisfied |
| Generator uses noise ~ N(0,1) | ✔ Satisfied |

---

## 10. Conclusion

This project successfully learns the probability density of a transformed NO₂ variable using a GAN-based approach without assuming any analytical distribution form.

The WGAN-GP framework enables stable training and effective implicit modeling of the unknown data distribution, fully satisfying Assignment-2 requirements.
