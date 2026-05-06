import torch


class EDMLoss:
    """
    EDM-style denoising loss for conditional time-series generation.

    Adapted from Karras et al. 2022 (Elucidating the Design Space of
    Diffusion-Based Generative Models, https://arxiv.org/abs/2206.00364)
    for the (B, K, L) conditional CSDI backbone.

    Differences from the image-generation reference:
    - Tensor layout is (B, K, L) not (B, C, H, W).
    - No augmentation pipe.
    - The model is called with the full CSDI-EDM conditional interface:
        model(x_t, sigma, observed_data, cond_mask, observed_tp) → D_x
    - Noise is added to all K channels of observed_data. Internally,
      CSDIModel.make_diff_input extracts only the target channel (Close)
      as the noisy input via (1 - cond_mask)*x_t; conditioning channels
      (O/H/L) are overwritten with clean data via cond_mask*observed_data.
    - Loss is applied only on target_mask entries (Close channel in
      predict_close mode), matching the masked_mse convention.

    Limitation (Phase 3):
    - The backbone (diff_CSDI) has not yet been retrained against this
      loss. Calling this loss during training will drive the backbone
      toward the correct EDM F_x target, but a model checkpoint trained
      with the old epsilon-MSE loss is not a valid warm start under the
      EDM objective. A fresh training run is required.
    """

    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 1.0):
        """
        Args:
            P_mean:     Mean of the log-normal sigma-sampling distribution.
                        ln sigma ~ N(P_mean, P_std^2).
                        EDM paper default: -1.2 (tuned for sigma_data=0.5, ImageNet).
                        With sigma_data=1.0 and sigma range [0.01, 5.0] this default
                        places the distribution mode at exp(-1.2) ≈ 0.30 and the 5th–95th
                        percentile at roughly [0.03, 3.2], covering the full useful sigma range.
                        Should be recalibrated empirically to the actual data distribution.
            P_std:      Std of the log-normal distribution. EDM default: 1.2.
            sigma_data: Empirical std of the (window-normalised) training data.
                        MUST match the sigma_data in CSDIModel preconditioning (Phase 1),
                        i.e. config["model"]["sigma_data"].
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: torch.nn.Module,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        target_mask: torch.Tensor,
        observed_tp: torch.Tensor,
        feature_id=None,
        debug: bool = False,
    ):
        """
        Compute the EDM training loss for one batch.

        Args:
            model:         CSDIModel with the Phase-1 EDM forward interface.
            observed_data: (B, K, L)  clean OHLC data (the denoising target).
            cond_mask:     (B, K, L)  1 = conditioning channel (O/H/L), 0 = target (Close).
            target_mask:   (B, K, L)  1 = entries included in the loss (Close only).
            observed_tp:   (B, L)     absolute time-series positions.
            feature_id:    optional feature indices forwarded to model.forward().
            debug:         if True, print per-step diagnostic statistics.

        Returns:
            loss:   scalar Tensor  — EDM-weighted MSE on target_mask entries.
                                    Ready for .backward().
            sigma:  (B,) Tensor   — per-sample noise levels used this step (for logging).
        """
        B = observed_data.shape[0]
        device = observed_data.device

        # --- 1. Sample sigma from log-normal: ln sigma ~ N(P_mean, P_std^2) ---
        # One independent draw per batch element (Karras et al. 2022, Sec. 5).
        # This distribution concentrates training on the perceptually important
        # noise levels and replaces the uniform-t sampling used in the old SDE path.
        rnd = torch.randn(B, device=device)                  # (B,)
        sigma = (rnd * self.P_std + self.P_mean).exp()       # (B,)  all > 0 by construction

        # --- 2. Construct noisy input: x_t = x_0 + sigma * eps ---
        # Broadcast sigma over K and L dimensions.
        # Only the Close channel (target) is consumed as a noisy input inside
        # model.forward(); O/H/L (conditioning) are always read from the clean
        # observed_data via cond_mask inside CSDIModel.make_diff_input.
        sigma_b = sigma.view(B, 1, 1)                        # (B, 1, 1)
        eps = torch.randn_like(observed_data)                # (B, K, L)
        x_t = observed_data + sigma_b * eps                  # (B, K, L)

        # --- 3. EDM weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2 ---
        # Karras et al. 2022, Eq. 9. Equivalently: 1/sigma_data^2 + 1/sigma^2.
        # Down-weights very high sigma (low-frequency, easy denoising) and diverges
        # at sigma → 0 — but the log-normal sampling keeps sigma bounded away from zero.
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2  # (B,)
        weight = weight.view(B, 1, 1)                        # (B, 1, 1) for (B, K, L) broadcast

        # --- 4. Call model: D_x = c_skip(sigma)*x_t + c_out(sigma)*F_x ---
        # The preconditioning (Phase 1) is applied internally by CSDIModel.forward().
        # D_x approximates E[x_0 | x_t, sigma] — the MMSE denoised estimate.
        D_x = model(
            x_t=x_t,
            sigma=sigma,
            observed_data=observed_data,
            cond_mask=cond_mask,
            observed_tp=observed_tp,
            feature_id=feature_id,
        )  # (B, K, L)

        # --- 5. Weighted MSE between D_x and the clean target, on target_mask only ---
        # The loss measures how well D_x reconstructs x_0 = observed_data.
        # Contributions from O/H/L (where target_mask == 0) are zeroed out.
        sq_err = (D_x - observed_data) ** 2                  # (B, K, L)

        # if debug:
            # masked_err = sq_err[target_mask.bool()]
            # print(
            #     f"EDM | sigma min={sigma.min():.4f}  max={sigma.max():.4f}  "
            #     f"mean={sigma.mean():.4f}  "
            #     f"sq_err(target) mean={masked_err.mean():.6f}  "
            #     f"weight mean={weight.mean():.4f}"
            # )
        # Weighted mean: each entry's squared error is weighted by lambda(sigma) of its sample.
        # The denominator sums lambda(sigma_b) * target_mask to normalise correctly.
        denom = (target_mask.float() * weight).sum().clamp(min=1e-10)
        loss = (sq_err * weight * target_mask).sum() / denom

        return loss, sigma, x_t
