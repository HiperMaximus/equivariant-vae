# Equivariant + Denoising VAEs for Histopathology (32×32)

> **Experiments repo for a paper** (not a library).  
> Goal: learn robust patch embeddings from RGB histopathology images (32×32).  
> Primary embedding target: **4×4×24** (384 dims).  
> Comparison of interest: **DVAE** (denoising VAE) vs **EQ-DVAE** (denoising + D4 equivariance).

---

## 1) Project summary (what & why)

**What.** We train VAEs that reconstruct histopathology patches while learning low-dimensional embeddings for downstream classification. We compare a **Denoising VAE (DVAE)** against an **Equivariant Denoising VAE (EQ-DVAE)**.

**Why.** Patches exhibit rotations/flips, micro-texture, and stain variability. A **denoising** criterion promotes robustness to local corruptions; **equivariance** regularizes representations to transform predictably under the dihedral group **D4** (90° rotations + flips). Together, these should yield embeddings that generalize better, especially with few labels or across sites.

**Deliverables.** Reproducible configs + scripts, exported embeddings for linear probes, and a report comparing DVAE vs EQ-DVAE.

---

## 2) Research questions & hypotheses

- **RQ1 (Equivariance):** Do EQ-DVAE embeddings improve linear-probe performance vs DVAE on patch-level labels?  
  **H1:** EQ-DVAE ≥ DVAE on Accuracy/AUROC, with larger gains in low-label and cross-site settings.

- **RQ2 (Robustness via denoising):** How much benefit comes from denoising alone vs denoising + equivariance?  
  **H2:** DVAE ≥ vanilla VAE; EQ-DVAE ≥ DVAE under corruptions and distribution shifts.

- **RQ3 (Group choice):** **(planned)** Compare D4 vs C4.

---

## 3) Method overview (how it works)

- **Denoising VAE (DVAE).** The encoder sees a corrupted input $\tilde{x}\sim q(\tilde{x}\mid x)$ (Gaussian noise, mild blur or JPEG). The decoder reconstructs the clean $x$. Objective is the standard ELBO with $\tilde{x}$ in the inference path.

- **Equivariant DVAE (EQ-DVAE).** Same denoising criterion plus an **equivariance consistency** regularizer to the **D4** group; we start with feature-space consistency under transformed inputs. **(Further equivariant layers / weight-tying: planned)**

---

## 4) Architecture (normalization-free)

**Encoder/decoder:** stacks of **pre-activation residual MBConv-style blocks** with **linear residual tails** and **linear residual adds**. No batch/instance/group normalization. Stabilization via **ReZero gates**, **Fixup-style scaling**, and selective **Scaled Weight Standardization (SWS)**.

### 4.1 Design principles

- **Pre-activation residual blocks:** activations live *inside* the branch; the final conv **and** the residual add are **linear** → identity + small residual without normalization.
- **Residual gating (ReZero/SkipInit):** per-block scalar $\alpha$, **initialized to 0**.
- **Fixup-style scaling:** scale branch weights so updates remain small across depth.
- **SWS:** apply only to **mixing** convs (stem, 1×1 PWs, and any full 3×3 convs). **No SWS** on depthwise convs.
- **Clean skip path:** identity where possible; otherwise a **linear** 1×1 projection (SWS), no activation.
- **Activation:** **SiLU** everywhere except **no activation** at the residual tail or around the add.

> **Init used here:** **ReZero mode** → $\alpha=0$ at init; **do not** zero-init the last PW. The last PW is **scaled with the Fixup factor** like the other branch weights (treat “as if it wasn’t the last”). Avoid double-damping ($\alpha=0$ **and** last-PW=0).

### 4.2 Encoder

- **Stem (mixing):** Conv 5×5, stride 1, C=16, SWS → SiLU.
- **ResidualBlock(C)** (pre-activation MBConv):
  1) Bias (0-init) → SiLU  
  2) **PW expand** 1×1 to 2C (**mixing**, SWS)  
  3) **DW** k×k, stride 1 (**no SWS**)  
  4) Bias (0-init) → SiLU  
  5) **PW project** 1×1 back to C (**mixing**, SWS, **linear tail**)  
  6) Multiply by **$\alpha$** (learnable; init 0)  
  7) Add clean skip (linear)
- **Downsample transition:** **PW expand ×2** (SWS) → **DW stride-2** (no SWS). Channels double when spatial halves.

**Default encoder schedule**
- Stem: 32×32×3 → 32×32×**16**  
- Stage 1 (×2 blocks @ 32×32): C=16  
- Downsample: 32→16; C: 16→**32**  
- Stage 2 (×2 blocks @ 16×16): C=32  
- Downsample: 16→8; C: 32→**64**  
- Stage 3 (×2 blocks @ 8×8): C=64  
- Downsample: 8→4; C: 64→**128**  
- **Head:** PW to **2×24** → interpret as $(\mu,\log\sigma^2)$ with shape $(B,24,4,4)$.

**Kernel note:** DW k=3 throughout (k=5 in Stage 1 is optional).

### 4.3 Latent

- Reparameterization: $z = \mu + \sigma \odot \epsilon$, with $z \in \mathbb{R}^{24\times4\times4}$.  
- Prior: standard isotropic Gaussian.

### 4.4 Decoder (symmetric, norm-free)

- **Upsampling:** nearest-neighbor ×2 + **mixing** 3×3 (SWS), preferred over transpose-conv to reduce checkerboard artifacts on 32×32.  
- **Stages:** mirror encoder (two-block stages at 4→8→16→32) with linear tails/adds.  
- **Output:** PW to 3 channels + **Sigmoid**.

---

## 5) Objectives & losses

- **Reconstruction:** $\mathcal{L}_{\text{rec}} = \|x - \hat{x}\|_2^2$  **(SSIM term: planned)**  
- **KL:** $\mathcal{L}_{\text{kl}} = \mathrm{KL}\!\big(q(z\mid x)\,\|\,\mathcal{N}(0,I)\big)$  
- **Equivariance (feature-space) to D4:**  
  $$
  \mathcal{L}_{\text{eq}} \;=\; \mathbb{E}_{g\sim \mathcal{G}}
  \left\| \, \phi(F(gx)) \;-\; \rho(g)\,\phi(F(x)) \, \right\|_2^2
  $$
  where $\phi$ extracts latent/features (e.g., $\mu$ or a pre-head map) and $\rho(g)$ is the induced representation (spatial transform and/or channel permutation).

- **DVAE (baseline):**
  $$
  \mathcal{L}_{\text{DVAE}} = \mathcal{L}_{\text{rec}} + \beta\,\mathcal{L}_{\text{kl}}
  $$
  Encoder sees $\tilde{x}\sim q(\tilde{x}\mid x)$ (noise/blur/jpeg); decoder targets clean $x$.

- **EQ-DVAE (proposed):**
  $$
  \mathcal{L}_{\text{EQ-DVAE}} = \mathcal{L}_{\text{rec}} + \beta\,\mathcal{L}_{\text{kl}} + \lambda_{\text{eq}}\,\mathcal{L}_{\text{eq}}
  $$

**Defaults to tune:** $\beta \in \{1,2,4\}$, $\lambda_{\text{eq}} \in [0.1,1.0]$.  
**Corruptions $q(\tilde{x}\mid x)$:** Gaussian $\sigma\in\{0.05,0.1\}$, light blur, or low-quality JPEG. Apply **only** to the encoder input; compute $\mathcal{L}_{\text{eq}}$ consistently (same corruption on both branches).

---

## 6) Data & preprocessing

- **Input:** RGB 32×32 patches from WSI (train/val/test).  
- **Layout:**
data/raw/
train/<class>/.png
val/<class>/.png
test/<class>/*.png

- **Transforms:** D4-closed flips/rotations; optional mild color jitter.  
- **Stain normalization:** **(planned)**

---

## 7) Evaluation protocols **(planned)**

- Reconstruction metrics (MSE/PSNR/SSIM), equivariance error, linear probe / small MLP, data-efficiency (1/5/10%), cross-site transfer, and efficiency (params/FLOPs/throughput).

---

## 8) Experiments & ablations **(planned)**

- DVAE vs EQ-DVAE; D4 vs C4; upsampling method; β sweep; noise types; stain norm on/off.  
- Report mean±std over ≥3 seeds.

---

## 9) Reproducibility

- Fixed seeds; PyTorch 2.8; Python ≥3.11; deterministic flags where feasible.  
- Log config + git commit; save checkpoints and exported embeddings under `data/processed/`.

---

## 10) Repo layout **(planned)**


---

## 11) Environment & installation (dev only)

> Uses **uv** for env/deps. PyTorch installed separately to match CUDA/MPS/CPU.

```bash
# create env
uv venv --python 3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install PyTorch (choose one)
# CPU:
uv pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0
# CUDA example (cu121):
# uv pip install --index-url https://download.pytorch.org/whl/cu121 \
#     torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0

# project deps (experiments only; not a library)
uv pip install -r requirements.txt
