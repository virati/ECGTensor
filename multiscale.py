#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiscale pre-mappings for ECG phase portraits.

The raw V1/V2 phase portrait is dominated by the QRS loop: P and T waves
collapse into a blob at the origin. Each mapping here rebalances the scales
before plotting:

    (a) signed_log      - sign(x) * log10(1 + |x|/x0), compresses QRS, expands P/T
    (b) wavelet2d       - beats stacked into a (beat x time) image per lead,
                          2D DWT, cross-beat denoise + per-level gain equalization
    (c) fractional      - Grunwald-Letnikov fractional integral/derivative
                          (power-law, scale-free kernel)
    (d) matryoshka      - autoencoder whose latent prefixes z[:m] are each trained
                          to reconstruct, so z[:2] is the best 2D portrait

@author: virati
"""
import numpy as np
import scipy.signal as sig
import wfdb

LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'I', 'II', 'III']


def load_ecg(record='focus_data/007c', fs_out=250, band=(0.5, 40.0)):
    """Load a STAFFIII record, band-pass to kill baseline wander, downsample.

    Returns (signal [n_samples x 9 leads], fs).
    """
    rec = wfdb.rdrecord(record)
    x = rec.p_signal
    sos = sig.butter(4, band, btype='bandpass', fs=rec.fs, output='sos')
    x = sig.sosfiltfilt(sos, x, axis=0)
    q = int(rec.fs // fs_out)
    x = sig.decimate(x, q, axis=0, zero_phase=True)
    return x, rec.fs / q


def find_r_peaks(x, fs):
    """R peaks from the multi-lead energy envelope."""
    env = np.linalg.norm(np.gradient(x, axis=0), axis=1)
    env = sig.sosfiltfilt(sig.butter(2, 15, fs=fs, output='sos'), env)
    peaks, _ = sig.find_peaks(env, distance=int(0.3 * fs),
                              height=np.percentile(env, 98) * 0.4)
    # Snap to the max-|amplitude| sample nearby
    w = int(0.04 * fs)
    snapped = [p - w + np.argmax(np.abs(x[max(p - w, 0):p + w]).sum(axis=1))
               for p in peaks]
    return np.array(snapped)


# ---------------------------------------------------------------- (a) log10
def signed_log(x, x0=None):
    """sign(x) * log10(1 + |x|/x0). x0 defaults to a per-lead robust scale
    (median absolute value), roughly the size of the P/T waves."""
    if x0 is None:
        x0 = np.median(np.abs(x), axis=0)
    return np.sign(x) * np.log10(1 + np.abs(x) / x0)


# ---------------------------------------------------------- (b) wavelet 2D
def beat_stack(x, fs, peaks, pre=0.25, post=0.55):
    """Stack beats into [n_beats x n_time x n_leads] aligned on R."""
    a, b = int(pre * fs), int(post * fs)
    peaks = peaks[(peaks - a >= 0) & (peaks + b <= len(x))]
    return np.stack([x[p - a:p + b] for p in peaks]), peaks


def wavelet2d(beats, wavelet='sym4', level=5, gamma=1.0, denoise=True):
    """Stationary (undecimated) 2D wavelet transform of each lead's
    (beat x time) image.

    - All detail bands are soft-thresholded (universal threshold from the
      finest diagonal band): denoising that exploits beat-to-beat consistency.
    - The within-beat time detail (cV) at level j (1 = finest) is gained by
      (2**j / 2**level)**gamma. gamma>0 attenuates the fine scales where the
      QRS lives, so P/T/ST open up relative to it; gamma=0 is plain denoising.

    The undecimated transform is shift-invariant, so re-weighting levels
    does not produce the blocky artifacts a decimated DWT would.
    """
    import pywt
    n_beats, n_t, n_ch = beats.shape
    m = 2 ** level
    pad_b, pad_t = -n_beats % m, -n_t % m
    out = np.empty_like(beats)
    for ch in range(n_ch):
        img = np.pad(beats[:, :, ch], ((0, pad_b), (0, pad_t)), mode='edge')
        # [cA_L, (cH,cV,cD)_L, ..., (cH,cV,cD)_1]; cH is high-pass along beats,
        # cV along within-beat time, cD both.
        coeffs = pywt.swt2(img, wavelet, level=level, trim_approx=True)
        sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(img.size))
        new = [coeffs[0]]
        for j, det in zip(range(level, 0, -1), coeffs[1:]):
            if denoise:
                det = [pywt.threshold(d, thr, 'soft') for d in det]
            g = (2.0 ** j / 2.0 ** level) ** gamma
            new.append((det[0], g * det[1], det[2]))
        out[:, :, ch] = pywt.iswt2(new, wavelet)[:n_beats, :n_t]
    return out


# ------------------------------------------------------- (c) fractional
def gl_fractional_derivative(x, alpha, fs=1.0, n_taps=256):
    """Grunwald-Letnikov fractional derivative of order alpha along axis 0.

    The GL kernel decays as a power law k^(-1-alpha): no characteristic time
    scale, so it treats all scales self-similarly. alpha=1 is the ordinary
    derivative (classic x vs dx/dt portrait, QRS explodes); alpha<0 is a
    fractional integral, a 1/f^|alpha| tilt that lifts the slow P/T waves
    relative to the fast QRS.
    """
    w = np.empty(n_taps)
    w[0] = 1.0
    for k in range(1, n_taps):
        w[k] = w[k - 1] * (1 - (alpha + 1) / k)
    y = sig.lfilter(w, [1.0], x, axis=0)
    return y * fs ** alpha


# ------------------------------------------------------- (d) Matryoshka AE
def _windows(x, win):
    half = win // 2
    xp = np.pad(x, ((half, half), (0, 0)), mode='edge')
    idx = np.arange(len(x))[:, None] + np.arange(win)[None, :]
    return xp[idx].reshape(len(x), -1)  # [n x (win*leads)]


def matryoshka_embed(x, win=32, latent=16, nest=(2, 3, 4, 8, 16),
                     steps=3000, batch=512, lr=2e-3, seed=0, verbose=False,
                     return_recon=False):
    """Matryoshka Representation Learning on sliding multi-lead windows.

    Encoder MLP -> z (latent dims). For every m in `nest`, z[:m] alone is
    decoded (shared MLP decoder, other dims zeroed) and its reconstruction
    loss is added. The early coordinates are forced to carry the coarsest,
    most explanatory structure, so (z0, z1) is a learned 2D phase plane and
    (z0, z1, z2) a 3D one. Returns z for every time sample [n x latent].
    With return_recon, also returns the reconstruction MSE (standardized
    units) when decoding from each prefix z[:m], m = 1..latent.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    X = _windows(x, win)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    d = Xt.shape[1]

    enc = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64),
                        nn.GELU(), nn.Linear(64, latent))
    dec = nn.Sequential(nn.Linear(latent, 64), nn.GELU(), nn.Linear(64, 128),
                        nn.GELU(), nn.Linear(128, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    masks = torch.stack([(torch.arange(latent) < m).float() for m in nest])

    for step in range(steps):
        xb = Xt[torch.randint(0, len(Xt), (batch,))]
        z = enc(xb)
        loss = sum(((dec(z * mk) - xb) ** 2).mean() for mk in masks) / len(nest)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 500 == 0:
            print(f'step {step}  loss {loss.item():.4f}')

    with torch.no_grad():
        z = enc(Xt)
        if return_recon:
            recon = np.array([((dec(z * (torch.arange(latent) < m).float())
                                - Xt) ** 2).mean().item()
                              for m in range(1, latent + 1)])
            return z.numpy(), recon
    return z.numpy()
