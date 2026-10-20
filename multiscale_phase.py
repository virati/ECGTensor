#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiscale pre-mappings for the V1/V2 phase portrait.

Every panel is colored by cardiac phase (fraction of the R-R interval), so
P / QRS / T segments line up across panels.

@author: virati
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import multiscale as ms

C1, C2 = 0, 1          # V1, V2
T_SPAN = 10.0          # seconds of continuous data to draw
T0 = 30.0              # start time (s)


def cardiac_phase(n, peaks):
    """Fraction of the current R-R interval, in [0, 1)."""
    ph = np.full(n, np.nan)
    for a, b in zip(peaks[:-1], peaks[1:]):
        ph[a:b] = np.arange(b - a) / (b - a)
    return ph


def phase_line(ax, x, y, c, lw=1.2, alpha=0.8):
    pts = np.stack([x, y], axis=1)[:, None, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='twilight_shifted', norm=plt.Normalize(0, 1),
                        linewidths=lw, alpha=alpha)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def style(ax, title, xl='V1', yl='V2'):
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.spines[['top', 'right']].set_visible(False)


if __name__ == '__main__':
    x, fs = ms.load_ecg('focus_data/007c')
    peaks = ms.find_r_peaks(x, fs)
    ph = cardiac_phase(len(x), peaks)
    seg = slice(int(T0 * fs), int((T0 + T_SPAN) * fs))
    c = ph[seg]

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.ravel()

    # Raw
    phase_line(axs[0], x[seg, C1], x[seg, C2], c)
    style(axs[0], 'Raw (band-passed, 250 Hz)')

    # (a) signed log10
    xl = ms.signed_log(x)
    phase_line(axs[1], xl[seg, C1], xl[seg, C2], c)
    style(axs[1], '(a) signed log10', 'slog V1', 'slog V2')

    # (b) 2D wavelet on the beat-stacked image
    beats, bpeaks = ms.beat_stack(x, fs, peaks)
    wb = ms.wavelet2d(beats, level=5, gamma=1.0)
    in_seg = (bpeaks >= seg.start) & (bpeaks < seg.stop)
    bph = np.linspace(-0.25, 0.55, beats.shape[1])  # time rel. R (s)
    bph = np.mod(bph / np.median(np.diff(peaks) / fs), 1.0)
    for b in np.where(in_seg)[0]:
        phase_line(axs[2], wb[b, :, C1], wb[b, :, C2], bph)
    style(axs[2], '(b) 2D wavelet (beat x time): denoise + scale EQ', 'V1', 'V2')

    # (c) fractional integral: power-law (scale-free) 1/f tilt
    alpha = -0.3
    xf = ms.gl_fractional_derivative(x, alpha, fs=fs)
    phase_line(axs[3], xf[seg, C1], xf[seg, C2], c)
    style(axs[3], rf'(c) fractional order $D^{{{alpha}}}$', rf'$D^{{{alpha}}}$V1',
          rf'$D^{{{alpha}}}$V2')

    # (d) Matryoshka AE latent prefix
    z = ms.matryoshka_embed(x)
    phase_line(axs[4], z[seg, 0], z[seg, 1], c)
    style(axs[4], '(d) Matryoshka AE, z[:2] (all 9 leads)', 'z0', 'z1')

    # (a)+(d): Matryoshka on the signed-log leads, so the loss isn't all QRS
    zl = ms.matryoshka_embed(xl)
    lc = phase_line(axs[5], zl[seg, 0], zl[seg, 1], c)
    style(axs[5], '(a)+(d) Matryoshka on signed-log leads', 'z0', 'z1')

    cb = fig.colorbar(lc, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label('cardiac phase (fraction of R-R)')
    plt.savefig('imgs/multiscale_phase.png', dpi=110, bbox_inches='tight')
    plt.show()
