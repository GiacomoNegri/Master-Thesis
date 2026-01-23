# Diffusion Models for Financial Time Series via Geometric Brownian Motion

Master’s thesis project on building a diffusion-based generative model that learns the implicit, non-parametric distribution of financial time series, with a focus on the S&P 500 index.

## Motivation

Classical econometric models such as ARIMA/GARCH are built around strong parametric assumptions (e.g. conditional Normal or Student-t returns) and struggle to reproduce several **stylized facts** of financial markets:

- Heavy-tailed return distributions  
- Volatility clustering  
- Asymmetric effects such as the leverage effect

GAN-based approaches improve flexibility but introduce training instability and mode collapse issues, which are undesirable for risk-sensitive applications. On the other hand, Diffusion models, and in particular score-based SDE formulations, offer a stable and expressive alternative: they model data by progressively noising and denoising samples, learning the full joint distribution without fixing a parametric form in advance.

---

## High-Level Idea

The core idea of this thesis is to **embed financial structure directly into the diffusion process**:

1. **Forward process – GBM-based noising**

   - Prices are first mapped to log-space.  
   - A **Geometric Brownian Motion (GBM)**, as in the Black–Scholes framework, is used as the forward SDE to inject noise in a *multiplicative* way, i.e. proportional to the asset level.  
   - By an appropriate choice of drift and diffusion terms, the forward SDE reduces to a variance-exploding process in log-prices, compatible with standard score-based diffusion / EDM training.

   This aims to reflect realistic **heteroskedasticity**: higher prices naturally carry higher volatility.

2. **Reverse process – CSDI Transformer based architecture within the EDM framework**

   - The reverse-time SDE is parameterized by a **Transformer denoiser** trained in the **Elucidated Diffusion Model (EDM)** framework, using noise-level preconditioning so that the network always operates on approximately unit-variance inputs.:contentReference[oaicite:5]{index=5} 
