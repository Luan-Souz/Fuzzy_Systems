# Fuzzy Systems Algorithms

A collection of fuzzy systems algorithms implemented during the Fuzzy Systems (Sistemas Nebulosos) graduate course at UEA (Universidade do Estado do Amazonas), as part of the Control and Automation Engineering undergraduate program.

## Table of Contents

- [1. Fuzzy Function Approximation](#1-fuzzy-function-approximation)
- [2. ANFIS – Stochastic Gradient Descent for Takagi-Sugeno Models](#2-anfis--stochastic-gradient-descent-for-takagi-sugeno-models)
- [3. Fuzzy C-Means Clustering and Image Segmentation](#3-fuzzy-c-means-clustering-and-image-segmentation)

---

## 1. Fuzzy Function Approximation

Approximation of the `sinc(x)` function over the interval `(0, π)` using a set of fuzzy rules with Gaussian membership functions and a weighted average defuzzification scheme.

### Approach

Seven Gaussian membership functions are defined across the input domain, each parameterized by a center `c` and a standard deviation `σ = 0.4`:

| Membership Function | Center (c) |
|---------------------|-----------|
| μ₁ | 0.00 |
| μ₂ | 0.65 |
| μ₃ | 1.30 |
| μ₄ | 1.95 |
| μ₅ | 2.50 |
| μ₆ | 2.85 |
| μ₇ | π |

### Fuzzy Rules (Takagi-Sugeno type)

Each rule maps a membership function to a consequent (either a constant or a linear function of `x`):

- **Rule 1:** If x is μ₁ then y₁ = 1
- **Rule 2:** If x is μ₂ then y₂ = −0.8512x + 1
- **Rule 3:** If x is μ₃ then y₃ = −0.2172
- **Rule 4:** If x is μ₄ then y₄ = 0.3355x − 0.6971
- **Rule 5:** If x is μ₅ then y₅ = 0.1284
- **Rule 6:** If x is μ₆ then y₆ = −0.2523x + 0.7491
- **Rule 7:** If x is μ₇ then y₇ = −0.0436

### Output Computation

The fuzzy approximation is computed via weighted average defuzzification:

$$y_{approx} = \frac{\sum_{i=1}^{7} y_i \cdot \mu_i}{\sum_{i=1}^{7} \mu_i}$$

### Results

- **Maximum absolute error:** 2.69 × 10⁻²
- **Mean Squared Error (MSE):** 2.1325 × 10⁻⁴

---

## 2. ANFIS – Stochastic Gradient Descent for Takagi-Sugeno Models

Implementation of the stochastic gradient descent (SGD) algorithm for parameter estimation of a Takagi-Sugeno fuzzy inference system. This is the core training mechanism behind ANFIS (Adaptive Neuro-Fuzzy Inference System).

### Model Structure

Given a dataset of samples `(Xᵢ, Yᵢ)` where `Xᵢ = [xᵢ₁, xᵢ₂, ..., xᵢd]ᵀ` and `Yᵢ` is a scalar output, the model is defined by `m` Takagi-Sugeno rules of the form:

**Rule k:** If x₁ is A₁ₖ and x₂ is A₂ₖ and ... and x_d is A_dk, then:

$$Y_k = p_{1k} x_1 + p_{2k} x_2 + \ldots + p_{dk} x_d + q_k$$

### Membership Functions

Gaussian membership functions are used for each antecedent:

$$\mu_{A_{\ell k}}(x_{i\ell}) = \exp\left(-\frac{1}{2} \frac{(x_{i\ell} - c_{\ell k})^2}{\sigma_{\ell k}^2}\right)$$

### Model Output

The output is a weighted average of rule consequents:

$$\hat{Y}_i = \frac{\sum_{k=1}^{m} w_k \cdot Y_k}{\sum_{k=1}^{m} w_k}$$

where the firing strength of each rule is:

$$w_k = \prod_{\ell=1}^{d} \mu_{A_{\ell k}}(x_{i\ell})$$

### SGD Parameter Update

The cost function minimized is:

$$J = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

Gradients are derived via chain rule for each parameter type — centers `cₗₖ`, dispersions `σₗₖ`, consequent weights `pₗₖ`, and biases `qₖ` — and updated using stochastic gradient descent (one sample per update step).

### Parameter Initialization

- **Centers (c):** Uniform random values within the range of the corresponding input variable
- **Dispersions (σ):** Uniform random values in `[0, 1]`
- **Consequent parameters (p, q):** Random values in `[-1, 1]`

Parameters are stored in three matrices (`C`, `σ`, `P` of dimension `d × m`) and one vector (`q` of dimension `m`).

---

## 3. Fuzzy C-Means Clustering and Image Segmentation

Implementation of the Fuzzy C-Means (FCM) algorithm for unsupervised data clustering and its application to RGB image segmentation. Results are benchmarked against MATLAB's built-in `fcm` function.

### Algorithm

FCM minimizes the following objective function:

$$J = \sum_{i=1}^{N} \sum_{j=1}^{C} \mu_{ij}^m \| x_i - c_j \|^2$$

where `N` is the number of data points, `C` is the number of clusters, `m` is the fuzzification coefficient, `μᵢⱼ` is the membership degree of point `xᵢ` in cluster `j`, and `cⱼ` is the center of cluster `j`.

**Membership update:**

$$\mu_{ij} = \left( \sum_{k=1}^{C} \left( \frac{\| x_i - c_j \|}{\| x_i - c_k \|} \right)^{\frac{2}{m-1}} \right)^{-1}$$

**Center update:**

$$c_j = \frac{\sum_{i=1}^{N} \mu_{ij}^m \cdot x_i}{\sum_{i=1}^{N} \mu_{ij}^m}$$

### Parameters Used

- Fuzzification coefficient: `m = 2`
- Convergence threshold: `ε = 10⁻³`
- Stopping criterion: `‖C⁽ᵗ⁾ − C⁽ᵗ⁻¹⁾‖ < ε`

### Experiments

1. **Data clustering:** Applied FCM to the `fcm_dataset.mat` dataset with 4 clusters, validating against MATLAB's `fcm` function.
2. **Image segmentation:** Adapted FCM for RGB image segmentation with varying cluster counts:
   - Image 1 (elephant): 4 clusters
   - Image 2 (aerial/satellite): 3 clusters
   - Image 3 (candy): 6 clusters

The implemented algorithm produced results consistent with MATLAB's reference implementation across all experiments.

---

## Tech Stack

- **MATLAB** — Primary implementation language for all algorithms
- **Python** — Alternative implementations for select algorithms

## References

- Coutinho, P. H. S. *Proposta de Novos Algoritmos Híbridos de Clusterização Fuzzy e suas aplicações.* Universidade Estadual de Santa Cruz, 2017.
- Cannon, R. L., Dave, J. V., & Bezdek, J. C. *Efficient implementation of the fuzzy c-means clustering algorithms.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986.

## Author

**Luan Ferreira de Souza**  

