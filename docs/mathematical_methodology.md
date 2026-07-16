# TBR Mathematical Methodology

This document provides a comprehensive overview of the mathematical foundations underlying Time-Based Regression (TBR) analysis. It is intended for researchers, data scientists, and practitioners who want to understand the statistical methodology implemented in this package.

---

## Table of Contents

1. [Overview](#overview)
2. [Statistical Model](#statistical-model)
3. [Prediction and Uncertainty](#prediction-and-uncertainty)
4. [Treatment Effect Estimation](#treatment-effect-estimation)
5. [Statistical Inference](#statistical-inference)
6. [Subinterval Analysis](#subinterval-analysis)
7. [Model Diagnostics](#model-diagnostics)
8. [Assumptions and Limitations](#assumptions-and-limitations)
9. [Notation Reference](#notation-reference)
10. [References](#references)

---

## Overview

### What is Time-Based Regression?

Time-Based Regression (TBR) is a statistical methodology for estimating causal treatment effects in before-after experimental designs. It addresses a fundamental question in causal inference: *What would have happened in the absence of the treatment?*

The core idea is straightforward:

1. **Pre-treatment period**: Establish a statistical relationship between a control group and a treatment group using linear regression
2. **Treatment period**: Use this relationship to predict *counterfactual* outcomes — what the treatment group would have experienced without the intervention
3. **Effect estimation**: Calculate the difference between observed outcomes and counterfactual predictions to estimate the treatment effect

TBR provides not just point estimates but complete statistical inference, including credible intervals and posterior probabilities for treatment effects.

### When to Use TBR

TBR is appropriate when you have:

- **Time series data** with measurements before and during a treatment period
- **Control and treatment groups** measured over the same time periods
- **A stable pre-treatment relationship** between control and treatment groups
- **No treatment contamination** in the control group during the test period

TBR applies to any domain where a before-after experiment is conducted with control and treatment groups observed over time.

TBR is particularly valuable when randomized controlled trials are impractical or when you need to measure cumulative effects over time.

### Key Advantages

**Statistical Rigor**
- Provides formal credible intervals with proper uncertainty quantification
- Accounts for both model uncertainty and residual variance
- Uses well-established regression theory with clear assumptions

**Interpretability**
- Results are expressed in the original units of measurement
- Cumulative effects show total impact over the test period
- Posterior probabilities give intuitive answers to practical questions

**Flexibility**
- Works with any time granularity (daily, weekly, hourly, etc.)
- Supports subinterval analysis for custom time windows
- Handles various data types and domains

**Transparency**
- All assumptions are explicit and testable
- Diagnostic tools help validate model appropriateness
- Mathematical foundations are fully documented

---

## Statistical Model

### The Linear Regression Framework

TBR models the relationship between the treatment group metric ($y_t$) and the control group metric ($x_t$) using simple linear regression:

$$
y_t = \beta_0 + \beta_1 x_t + \varepsilon_t
$$

where:
- $y_t$ is the treatment group metric at time $t$
- $x_t$ is the control group metric at time $t$
- $\beta_0$ is the intercept (baseline offset)
- $\beta_1$ is the slope (scaling factor between groups)
- $\varepsilon_t$ is the random error term

The model is fitted using data from the **pre-treatment period only**, where no intervention has occurred. This ensures that the estimated relationship reflects the natural correlation between groups, uncontaminated by treatment effects.

### Model Assumptions

The TBR methodology relies on standard linear regression assumptions:

**1. Linearity**

The relationship between the control and treatment group metrics is linear. This means that changes in the control group metric produce proportional changes in the expected treatment group metric.

**2. Independence**

The error terms $\varepsilon_t$ are independent across time periods:

$$
\text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \quad \text{for all } t \neq s
$$

**3. Homoscedasticity**

The error terms have constant variance across all time periods:

$$
\text{Var}(\varepsilon_t) = \sigma^2 \quad \text{for all } t
$$

**4. Normality**

The error terms follow a normal distribution:

$$
\varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

**5. Exogeneity**

The error terms have zero mean conditional on the control group values:

$$
\mathbb{E}[\varepsilon_t \mid x_t] = 0
$$

This ensures that the regression estimators are unbiased.

**6. Stable Relationship**

The relationship established during the pre-treatment period continues to hold during the treatment period. This is the key assumption enabling counterfactual prediction.

### Parameter Estimation (OLS)

The regression coefficients are estimated using Ordinary Least Squares (OLS) on the pre-treatment data. Given $n$ observations in the pre-treatment period:

**Sample Means**

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i, \quad \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

**Sum of Squared Deviations**

$$
S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

**Slope Estimator**

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{S_{xx}}
$$

**Intercept Estimator**

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
$$

**Residual Variance Estimator**

The variance of the error term is estimated from the residuals:

$$
s^2 = \frac{1}{n - 2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ is the fitted value. The denominator uses $n - 2$ degrees of freedom to provide an unbiased estimate of $\sigma^2$.

---

## Prediction and Uncertainty

Once the regression model has been fitted on the pre-treatment period, TBR uses it to generate **counterfactual predictions** — estimates of what the treatment group would have experienced in the absence of the intervention. Quantifying the uncertainty of these predictions is essential for valid statistical inference about treatment effects.

### Counterfactual Predictions

During the treatment period, the control group metric $x_t$ is observed but the treatment group is affected by the intervention. The counterfactual prediction for each time point $t$ in the treatment period is:

$$
\hat{y}_t^* = \hat{\beta}_0 + \hat{\beta}_1 x_t, \quad t \text{ in the test period}
$$

where:
- $\hat{y}_t^*$ is the predicted value the treatment group would have taken without treatment at time $t$
- $x_t$ is the observed control group metric at time $t$
- $\hat{\beta}_0$ and $\hat{\beta}_1$ are the regression coefficients estimated from the pre-treatment period

These predictions rely on the assumption that the relationship between the control and treatment groups, established during the pre-treatment period, continues to hold during the treatment period.

### Model Variance

The model variance captures the uncertainty in $\hat{y}_t^*$ that arises solely from estimating the regression coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$. If the true coefficients were known, this component would be zero.

$$
\mathbb{V}[\hat{y}_t^*] = s^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)
$$

where:
- $s^2$ is the estimated residual variance from the pre-treatment period
- $n$ is the number of pre-treatment observations
- $\bar{x}$ is the mean of the control group metric in the pre-treatment period
- $S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$ is the sum of squared deviations

The model variance is smallest when the control group value $x_t$ is close to the pre-treatment mean $\bar{x}$, and increases as $x_t$ moves further from $\bar{x}$. It also decreases with larger pre-treatment sample size $n$.

### Prediction Variance

The prediction variance captures the **total** uncertainty in a counterfactual observation $y_t^*$, combining both the model uncertainty and the irreducible residual noise:

$$
y_t^* = \hat{y}_t^* + \varepsilon_t^*, \quad \varepsilon_t^* \sim \mathcal{N}(0, \sigma^2)
$$

Since $\hat{y}_t^*$ and $\varepsilon_t^*$ are independent:

$$
\mathbb{V}[y_t^*] = \mathbb{V}[\hat{y}_t^*] + \sigma^2
$$

Substituting the model variance and replacing $\sigma^2$ with its estimator $s^2$:

$$
\mathbb{V}[y_t^*] = s^2 \left( 1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)
$$

The prediction standard deviation is then:

$$
\sqrt{\mathbb{V}[y_t^*]} = s \cdot \sqrt{1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}}}
$$

### Distinguishing Model Variance and Prediction Variance

These two quantities serve different purposes in the TBR framework:

| Quantity | Formula | Source of uncertainty | Role in TBR |
|----------|---------|----------------------|-------------|
| Model variance $\mathbb{V}[\hat{y}_t^*]$ | $s^2 \left( \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)$ | Estimation of regression coefficients | Cumulative effect variance and credible intervals |
| Prediction variance $\mathbb{V}[y_t^*]$ | $s^2 \left( 1 + \frac{1}{n} + \frac{(x_t - \bar{x})^2}{S_{xx}} \right)$ | Coefficient estimation **and** residual noise | Pointwise prediction intervals for counterfactual outcomes |

The prediction variance is always larger than the model variance by exactly $s^2$ (the residual variance term). In practice:

- **Model variance** ($\mathbb{V}[\hat{y}_t^*]$) is used when computing the cumulative effect variance and credible intervals for the treatment effect
- **Prediction variance** ($\mathbb{V}[y_t^*]$) is used for pointwise prediction intervals around individual counterfactual values

Both quantities increase as the control group value $x_t$ moves further from $\bar{x}$, reflecting greater extrapolation uncertainty.

---

## Treatment Effect Estimation

With counterfactual predictions in hand, TBR estimates the causal effect of the treatment by comparing observed outcomes to their counterfactual values. The methodology provides both pointwise (daily) and cumulative effect estimates, along with rigorous uncertainty quantification.

### Pointwise Treatment Effect

For each time point $t$ in the test period, the pointwise causal effect is the difference between the observed treatment group outcome and the counterfactual prediction:

$$
\phi_t = y_t - \hat{y}_t^*
$$

where:
- $y_t$ is the observed treatment group metric at time $t$
- $\hat{y}_t^* = \hat{\beta}_0 + \hat{\beta}_1 x_t$ is the counterfactual prediction

A positive $\phi_t$ indicates the treatment increased the metric at time $t$; a negative $\phi_t$ indicates a decrease.

During the pretest period, $\phi_t$ reduces to the model residuals and can be used as a visual diagnostic for model fit.

### Cumulative Treatment Effect

The primary quantity of interest in TBR is the cumulative causal effect over the test period, defined as the sum of the pointwise effects from the first day of the test period up to time $T$:

$$
\Delta(T) = \sum_{t=1}^{T} \phi_t = \sum_{t=1}^{T} (y_t - \hat{y}_t^*)
$$

This measures the total impact of the treatment over the test period.

### Posterior Variance of the Cumulative Effect

The uncertainty in $\Delta(T)$ arises from two independent sources:

1. **Residual noise**: each outcome $y_t$ in the test period follows $y_t = \beta_0 + \beta_1 x_t + \varepsilon_t$, where the noise $\varepsilon_t$ has variance $\sigma^2$
2. **Model uncertainty**: the counterfactual predictions $\hat{y}_t^*$ depend on the estimated coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$, which carry estimation uncertainty from the finite pretest sample

These two sources are independent — the residual noise in the test period is independent of the parameter uncertainty from the pretest model — so the posterior variance decomposes as:

$$
\mathbb{V}[\Delta(T)] = \mathbb{V}\left(\sum_{t=1}^{T} y_t\right) + \mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right)
$$

**Residual component.** Under the assumption of independent, identically distributed residuals with variance $\sigma^2$:

$$
\mathbb{V}\left(\sum_{t=1}^{T} y_t\right) = T \cdot \sigma^2
$$

**Model component.** The variance of the sum of counterfactual predictions is:

$$
\mathbb{V}\left(\sum_{t=1}^{T} \hat{y}_t^*\right) = T^2 \cdot v
$$

where:

$$
v = \mathbb{V}[\hat{\beta}_0] + 2\bar{x}_T \cdot \text{Cov}(\hat{\beta}_0, \hat{\beta}_1) + \bar{x}_T^2 \cdot \mathbb{V}[\hat{\beta}_1]
$$

and $\bar{x}_T = \frac{1}{T} \sum_{t=1}^{T} x_t$ is the mean of the control group metric during the test period.

**Combined posterior variance.** Putting both components together:

$$
\mathbb{V}[\Delta(T)] = T \cdot \sigma^2 + T^2 \cdot v
$$

The posterior standard deviation of the cumulative effect is then:

$$
\text{SE} = \sqrt{\mathbb{V}[\Delta(T)]} = \sqrt{T \cdot \sigma^2 + T^2 \cdot v}
$$

This quantity grows over time as both variance components accumulate. The residual component grows linearly with $T$, while the model component grows quadratically with $T$, reflecting the compounding uncertainty from using estimated coefficients across an increasing number of time points.

---

## Statistical Inference

TBR uses the posterior distribution of the cumulative treatment effect to provide formal statistical inference, including credible intervals and posterior probabilities. This section describes the inferential framework.

### The t-Distribution Framework

Under the standard linear regression assumptions and a noninformative prior on $(\beta_0, \beta_1, \log \sigma)$, the posterior distribution of the cumulative treatment effect $\Delta(T)$ follows a Student's $t$-distribution:

$$
\Delta(T) \mid \text{data} \sim t_\nu\!\left(\hat{\Delta}(T),\; \text{SE}\right)
$$

where:
- $\hat{\Delta}(T) = \sum_{t=1}^{T} (y_t - \hat{y}_t^*)$ is the posterior mean (the point estimate)
- $\text{SE} = \sqrt{\mathbb{V}[\Delta(T)]}$ is the posterior standard deviation (scale of the distribution)
- $\nu = n - 2$ is the degrees of freedom, with $n$ being the number of pretest observations

The use of the $t$-distribution (rather than a normal distribution) accounts for the additional uncertainty from estimating $\sigma^2$ with a finite sample. As $\nu$ increases, the $t$-distribution converges to the normal distribution.

### Degrees of Freedom

The degrees of freedom $\nu$ are determined by the residual degrees of freedom from the regression model fitted on the pretest period:

$$
\nu = n - p
$$

where:
- $n$ is the number of observations in the pretest period
- $p$ is the number of estimated model parameters

For simple linear regression (intercept and slope), $p = 2$, giving $\nu = n - 2$.

A larger pretest period yields more degrees of freedom, which narrows the $t$-distribution and produces tighter credible intervals. For example, with $n = 90$ pretest observations, $\nu = 88$.

### Credible Intervals

Given the posterior $t$-distribution, TBR constructs a symmetric two-sided credible interval at confidence level $1 - \alpha$:

$$
[\text{lower},\; \text{upper}] = \hat{\Delta}(T) \pm t_{\alpha/2,\;\nu} \cdot \text{SE}
$$

where:
- $t_{\alpha/2,\;\nu}$ is the critical value from the Student's $t$-distribution with $\nu$ degrees of freedom
- $\alpha = 1 - \text{level}$; for example, $\alpha = 0.10$ for a 90% credible interval

The half-width of the credible interval (referred to as the **precision**) is:

$$
\text{precision} = t_{\alpha/2,\;\nu} \cdot \text{SE}
$$

This interval represents the range in which the true cumulative treatment effect $\Delta(T)$ lies with posterior probability $1 - \alpha$, given the model and observed data.

### Posterior Probability

TBR also computes the posterior probability that the true cumulative treatment effect exceeds a specified threshold $\theta$:

$$
P(\Delta(T) > \theta \mid \text{data})
$$

This is evaluated using the cumulative distribution function of the posterior $t$-distribution:

$$
P(\Delta(T) > \theta \mid \text{data}) = 1 - F_{t_\nu}\!\left(\frac{\theta - \hat{\Delta}(T)}{\text{SE}}\right)
$$

where $F_{t_\nu}$ is the CDF of the Student's $t$-distribution with $\nu$ degrees of freedom.

When $\theta = 0$, this gives the posterior probability that the treatment had a positive effect. This provides an intuitive, decision-relevant summary: for example, "there is a 95% posterior probability that the treatment increased the response metric."

---

## Subinterval Analysis

TBR supports analysis of treatment effects over arbitrary subintervals of the test period. Rather than restricting inference to the full test period, subinterval analysis allows estimation of cumulative effects for custom time windows — for example, examining only the first week of treatment, or a specific phase within a longer experiment.

### Approach

To analyze a subinterval from day $a$ to day $b$ within the test period, the analysis redefines the summation bounds while keeping the original pretest regression model unchanged. The regression parameters ($\hat{\beta}_0$, $\hat{\beta}_1$, $s^2$, $\nu$) remain the same, since the underlying model is always trained on the full pretest period.

The cumulative treatment effect for the subinterval is:

$$
\Delta(a, b) = \sum_{t=a}^{b} (y_t - \hat{y}_t^*)
$$

### Posterior Variance for Subintervals

The posterior variance of the subinterval effect combines the same two components as the full test period — model uncertainty and residual noise — but restricted to the days in the subinterval.

Let $T_s = b - a + 1$ be the number of days in the subinterval. The posterior variance is:

$$
\mathbb{V}[\Delta(a, b)] = \sum_{t=a}^{b} \mathbb{V}[\hat{y}_t^*] + T_s \cdot \sigma^2
$$

where $\mathbb{V}[\hat{y}_t^*]$ is the model variance at each time point (as defined in the [Prediction and Uncertainty](#prediction-and-uncertainty) section).

The posterior standard error is:

$$
\text{SE}_{(a,b)} = \sqrt{\mathbb{V}[\Delta(a, b)]}
$$

### Credible Intervals for Subintervals

The credible interval for the subinterval effect follows the same $t$-distribution framework as the full test period, using the same degrees of freedom $\nu$ from the pretest regression:

$$
[\text{lower},\; \text{upper}] = \Delta(a, b) \pm t_{\alpha/2,\;\nu} \cdot \text{SE}_{(a,b)}
$$

This approach is valid because the regression model is independent of which portion of the test period is being analyzed. The pretest relationship does not change; only the time window over which the treatment effect is accumulated changes.

---

## Model Diagnostics

The validity of TBR inference depends on the regression assumptions holding for the pretest data. This section describes the diagnostic tools available for assessing model adequacy.

### Residual Analysis

The pretest residuals are the differences between observed and fitted values:

$$
e_i = y_i - \hat{y}_i = y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i), \quad i = 1, \ldots, n
$$

If the model is well-specified, the residuals should behave as a random sample from a distribution with zero mean and constant variance. TBR provides three forms of residuals for diagnostic purposes:

- **Raw residuals** ($e_i$): the differences above, in the original units of the response
- **Standardized residuals** ($e_i / s$): scaled by the residual standard deviation, useful for identifying outliers (values beyond $\pm 2$ are noteworthy)
- **Studentized residuals** ($e_i / (s\sqrt{1 - h_{ii}})$): adjusted for the leverage $h_{ii}$ of each observation, providing more accurate outlier detection

The leverage $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H = X(X^\top X)^{-1}X^\top$, where $X$ is the $n \times 2$ design matrix with a column of ones and the pretest control values. For simple linear regression it reduces to:

$$
h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{S_{xx}}
$$

Observations with high leverage exert greater influence on the fitted regression line, which can mask outliers in raw and standardized residuals. Studentized residuals correct for this effect. Note that studentized residuals require $h_{ii} < 1$.

### Assumption Checking

TBR implements formal tests for each of the key regression assumptions:

**Normality**

The Shapiro-Wilk test assesses whether the residuals follow a normal distribution. This assumption underpins the $t$-distribution framework used for credible intervals and posterior probabilities. Moderate departures from normality have limited impact when $n$ is large, due to the central limit theorem.

**Homoscedasticity**

The Breusch-Pagan test checks whether the residual variance is constant across different levels of the predictor $x_t$. Violation (heteroscedasticity) can lead to incorrect standard errors and unreliable credible intervals.

**Independence**

The Durbin-Watson test detects first-order autocorrelation in the residuals. The test statistic ranges from 0 to 4:
- Values near 2 indicate no autocorrelation
- Values below 1.5 suggest positive autocorrelation
- Values above 2.5 suggest negative autocorrelation

Autocorrelation violates the independence assumption and can cause the posterior variance to be underestimated, leading to credible intervals that are too narrow.

### Goodness-of-Fit Metrics

TBR reports several metrics that summarize how well the pretest regression model fits the data:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| $R^2$ | $1 - SS_{\text{res}} / SS_{\text{tot}}$ | Proportion of variance in $y$ explained by the model |
| Adjusted $R^2$ | $1 - (1 - R^2) \cdot \frac{n-1}{n-k-1}$ | $R^2$ corrected for the number of predictors |
| $F$-statistic | $\frac{SS_{\text{reg}} / k}{SS_{\text{res}} / (n - k - 1)}$ | Tests overall model significance |
| RMSE | $\sqrt{\frac{1}{n-2} \sum e_i^2}$ | Residual standard deviation in original units |

where $SS_{\text{tot}} = \sum(y_i - \bar{y})^2$, $SS_{\text{res}} = \sum e_i^2$, $SS_{\text{reg}} = SS_{\text{tot}} - SS_{\text{res}}$, and $k$ is the number of predictors (typically 1 for simple linear regression).

A high $R^2$ indicates a strong pretest relationship between the control and treatment groups, which improves the precision of counterfactual predictions. A statistically significant $F$-statistic confirms that the linear relationship is meaningful.

---

## Assumptions and Limitations

### When TBR is Appropriate

TBR is designed for before-after experimental designs where:

- A **pretest period** establishes the baseline relationship between control and treatment groups
- A **test period** introduces an intervention whose causal effect is to be estimated
- The **control group** remains unaffected by the intervention throughout the experiment
- The response metric is observed as a **time series** at regular intervals

### Key Assumptions

The validity of TBR inference rests on the following assumptions:

**1. Linearity**

The relationship between the control and treatment group metrics is linear. If the true relationship is nonlinear, the counterfactual predictions will be biased, leading to biased treatment effect estimates.

**2. Stable relationship**

The pretest relationship between groups persists unchanged during the test period. This is the central assumption enabling counterfactual prediction. It can be violated by external events that differentially affect the treatment and control groups.

**3. No treatment contamination**

The control group is not affected by the intervention. If the treatment "spills over" into the control group, the counterfactual predictions will not reflect a true no-treatment baseline.

**4. Independent residuals**

The error terms $\varepsilon_t$ are independent across time. Autocorrelation in the residuals causes the posterior variance to be underestimated, producing credible intervals that are too narrow. The Durbin-Watson test (see [Model Diagnostics](#model-diagnostics)) can detect this violation.

**5. Homoscedasticity**

The residual variance is constant across all time periods. Heteroscedasticity leads to inefficient estimates and unreliable standard errors.

**6. Normality**

The error terms follow a normal distribution. This assumption supports the $t$-distribution framework for credible intervals and posterior probabilities. With sufficient pretest observations, moderate departures from normality have limited impact due to the central limit theorem.

### Limitations

**Single predictor.** TBR uses simple linear regression with one predictor (the control group metric). It does not accommodate multiple control variables or more complex model structures.

**Sensitivity to pretest length.** Short pretest periods yield imprecise parameter estimates and wide credible intervals. Increasing $n$ improves precision, but with diminishing returns: the credible interval width decreases at a rate of $1/\sqrt{n}$ and is bounded below by the residual variance $\sigma^2$ (see {cite}`kerman2017tbr`, Section 9.3).

**No structural breaks.** TBR assumes the regression parameters ($\beta_0$, $\beta_1$, $\sigma^2$) remain constant throughout the pretest period. If the relationship between groups changes during the pretest period (e.g., due to a trend or regime shift), the fitted model may not generalize to the test period.

**Cumulative uncertainty growth.** The posterior variance of the cumulative effect grows with the test period length $T$ as $\mathbb{V}[\Delta(T)] = T \cdot \sigma^2 + T^2 \cdot v$ (see [Treatment Effect Estimation](#treatment-effect-estimation)). Long test periods accumulate substantial uncertainty, reducing the precision of the treatment effect estimate.

**i.i.d. residual approximation.** The variance formula $\mathbb{V}\!\left[\sum y_t\right] = T \cdot \sigma^2$ is exact only when the residuals $\varepsilon_t$ are independent with constant variance $\sigma^2$. When residuals are autocorrelated, it becomes an approximation that may underestimate the true variance. In practice, Kerman et al. {cite}`kerman2017tbr` demonstrate that TBR credible intervals achieve accurate coverage even under correlated conditions.

---

## Notation Reference

The following table maps the mathematical notation used in this document to the corresponding Python variable names in the TBR package.

### Model Parameters

| Symbol | Meaning | Python Variable |
|--------|---------|-----------------|
| $\beta_0$ | Regression intercept | `alpha` |
| $\beta_1$ | Regression slope | `beta` |
| $\sigma$ | Residual standard deviation | `sigma` |
| $s^2$ | Estimated residual variance (estimator of $\sigma^2$) | `sigma**2` |
| $\mathbb{V}[\hat{\beta}_0]$ | Variance of intercept estimate | `var_alpha` |
| $\mathbb{V}[\hat{\beta}_1]$ | Variance of slope estimate | `var_beta` |
| $\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$ | Covariance of intercept and slope | `cov_alpha_beta` |
| $\nu$ | Degrees of freedom | `degrees_freedom` |
| $n$ | Number of pretest observations | `n_pretest` |
| $\bar{x}$ | Mean of control in pretest period | `pretest_x_mean` |

### Daily Output Columns

| Symbol | Meaning | Column Name |
|--------|---------|-------------|
| $y_t$ | Treatment group metric at time $t$ | `y` |
| $x_t$ | Control group metric at time $t$ | `x` |
| $\hat{y}_t$ or $\hat{y}_t^*$ | Fitted value (pretest) or counterfactual prediction (test) | `pred` |
| $\sqrt{\mathbb{V}[y_t^*]}$ | Prediction standard deviation | `predsd` |
| $\phi_t = y_t - \hat{y}_t^*$ | Pointwise treatment effect | `dif` |
| $\sum \phi_t$ | Cumulative treatment effect | `cumdif` |
| $\sqrt{\mathbb{V}[\Delta(T)]}$ | Cumulative effect standard deviation | `cumsd` |
| $\sqrt{\mathbb{V}[\hat{y}_t^*]}$ | Model standard deviation (fitted value) | `estsd` |

### Summary Output

| Symbol | Meaning | Field Name |
|--------|---------|------------|
| $\hat{\Delta}(T)$ | Cumulative treatment effect estimate | `estimate` |
| $\text{SE}$ | Posterior standard deviation of cumulative effect | `se` |
| $t_{\alpha/2,\nu} \cdot \text{SE}$ | Half-width of credible interval | `precision` |
| $[\text{lower}, \text{upper}]$ | Credible interval bounds | `lower`, `upper` |
| $1 - \alpha$ | Credible interval level | `level` |
| $\theta$ | Threshold for posterior probability | `thres` |
| $P(\Delta(T) > \theta \mid \text{data})$ | Posterior probability of exceeding threshold | `prob` |

---

## References

```{bibliography}
:all:
```

---
