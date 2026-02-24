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

1. Assume z is sampled from an unknown distribution.
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

1. Download the dataset from Kaggle:
- https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data
  
Rename it and place in root project folder:
```
Data.csv
```

2. Run:

```bash
python gan_pdf_model.py
```

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

Input: epsilon ~ N(0, 1)

Output: z_f = G(epsilon)

---

### Discriminator 

Fully connected neural network:

- Linear(1 → 128)
- LeakyReLU(0.2)
- Linear(128 → 64)
- LeakyReLU(0.2)
- Linear(64 → 1)

The discriminator distinguishes between:

- Real samples: \( z \)
- Fake samples: z_f = G(epsilon)
  
---

## 7. PDF Learned by GAN

After training:

1. A large number of samples \( z_f \) are generated from the trained generator.
2. Kernel Density Estimation (KDE) is applied to approximate:

Estimated density p(z)

The figure below shows the learned probability density of the transformed variable z. 
The density is obtained from GAN-generated samples using Kernel Density Estimation (KDE) and is a **non-parametric density estimate**

<img width="729" height="577" alt="image" src="https://github.com/user-attachments/assets/d8ac3060-521e-4931-af46-6e8affc70656" />

---

## 8. Results

### Training Stability

The discriminator loss increases steadily while the generator loss becomes more negative and stabilizes over time. This behavior indicates that the Wasserstein objective is being optimized; however, the steadily rising critic loss suggests the discriminator may be dominating the generator during later training stages.

Overall, training remains numerically stable without divergence.

---

### Mode Coverage

The mode coverage score is approximately:

8.7 × 10⁻⁵

This very low score indicates that the generator captures only a small portion of the real distribution’s support. While it models the dominant density region, it underrepresents broader regions of the transformed distribution.

---

### Distribution Quality

The KDE-based PDF shows a sharp peak in the lower range of z with a gradually decaying tail. The generator successfully captures the general skewed structure of the distribution but fails to fully reproduce the spread and tail behavior of the real data.

This suggests partial learning of the distribution, with limited diversity in generated samples.

---

## 9. Conclusion

The GAN successfully learned a non-parametric approximation of the transformed NO₂ distribution without assuming any analytical form. 

While training remained stable under the WGAN-GP framework, the low mode coverage score indicates incomplete distribution coverage. The generator captures the primary density region but does not fully model the variability and tail behavior of the data.

Further improvements could include longer training, architectural tuning, or hyperparameter optimization to enhance mode coverage and distribution quality.
