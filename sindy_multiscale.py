#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn SINDy models of the ECG in each multiscale representation.

For every representation (raw, signed log, 2D wavelet, fractional order,
Matryoshka, log + Matryoshka) we take a 3D state (V1/V2/V3 or z0/z1/z2),
split it into beats, and fit three sparse models:

    autonomous        x' = f(x)                    cubic polynomial library
    forced linear     x' = A x + B u(θ)            u = cos kθ, sin kθ, k = 1..6
    forced quadratic  x' = f2(x, u(θ))             quadratic library on [x, u]

θ is the cardiac phase (fraction of the R-R interval). The forced models get
the cardiac clock as an input (SINDYc), so they only have to learn how the
waveform responds to the phase, not generate the rhythm themselves.

Time is measured in beats (t / median R-R), so one cycle has period ~1 and
derivative magnitudes are comparable across representations.

Train: beats from 20-90 s. Test: beats from 90-110 s.
Metrics on test beats: derivative R^2, one-beat rollout NRMSE, model size.

@author: virati
"""
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pysindy as ps

import multiscale as ms

TRAIN = (20.0, 90.0)
TEST = (90.0, 110.0)
THRESHOLDS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
HARMONICS = 6
MODELS = {  # name: (polynomial degree, forced)
    'autonomous': (3, False),
    'forced linear': (1, True),
    'forced quadratic': (2, True),
}
DIVERGED = 10.0  # NRMSE reported for a rollout that blows up


def split_beats(y, peaks, lo, hi, fs):
    """R-to-R trajectories with cardiac phase, for beats starting in [lo, hi) s."""
    out = []
    for p, q in zip(peaks[:-1], peaks[1:]):
        if lo * fs <= p < hi * fs:
            out.append((y[p:q + 1, :3], np.linspace(0, 1, q - p + 1)))
    return out


def wavelet_beats(x, fs, peaks, lo, hi):
    beats, bp = ms.beat_stack(x, fs, peaks)
    wb = ms.wavelet2d(beats)
    rr = np.median(np.diff(peaks)) / fs
    ph = np.mod(np.linspace(-0.25, 0.55, beats.shape[1]) / rr, 1.0)
    return [(wb[i, :, :3], ph) for i in np.where((bp >= lo * fs) & (bp < hi * fs))[0]]


def build_representations(x, fs, peaks):
    xl = ms.signed_log(x)
    cont = {
        'Raw': x,
        'Signed log10': xl,
        'Fractional D^-0.3': ms.gl_fractional_derivative(x, -0.3, fs=fs),
        'Matryoshka': ms.matryoshka_embed(x),
        'Log + Matryoshka': ms.matryoshka_embed(xl),
    }
    reps = {}
    for name, y in cont.items():
        reps[name] = {s: split_beats(y, peaks, *rng, fs) for s, rng in (('train', TRAIN), ('test', TEST))}
    reps['2D wavelet'] = {s: wavelet_beats(x, fs, peaks, *rng) for s, rng in (('train', TRAIN), ('test', TEST))}
    order = ['Raw', 'Signed log10', '2D wavelet', 'Fractional D^-0.3', 'Matryoshka', 'Log + Matryoshka']
    return {k: reps[k] for k in order}


def normalize(rep):
    """Center and scale by the training set (one isotropic scale, shape preserved)."""
    allp = np.concatenate([b for b, _ in rep['train']])
    c, s = allp.mean(0), np.abs(allp - allp.mean(0)).max()
    return {k: [((b - c) / s, ph) for b, ph in v] for k, v in rep.items()}


def phase_input(ph):
    k = np.arange(1, HARMONICS + 1)
    a = 2 * np.pi * np.outer(ph, k)
    return np.column_stack([np.cos(a), np.sin(a)])


U_NAMES = [f'{f}{k}' for f in ('c', 's') for k in range(1, HARMONICS + 1)]


def make_model(threshold, degree):
    return ps.SINDy(optimizer=ps.STLSQ(threshold=threshold, alpha=1e-3),
                    feature_library=ps.PolynomialLibrary(degree=degree),
                    differentiation_method=ps.SmoothedFiniteDifference())


def fit(rep, degree, forced, dt, val_frac=0.2):
    """Sweep the sparsity threshold; keep the sparsest model whose validation
    derivative R^2 is within 0.02 of the best."""
    tr = rep['train']
    n_val = max(1, int(len(tr) * val_frac))
    fit_set, val_set = tr[:-n_val], tr[-n_val:]

    def args(beats):
        kw = dict(x=[b for b, _ in beats], t=dt)
        if forced:
            kw['u'] = [phase_input(ph) for _, ph in beats]
        return kw

    names = ['x', 'y', 'z'] + (U_NAMES if forced else [])
    cands = []
    for th in THRESHOLDS:
        m = make_model(th, degree)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(**args(fit_set), feature_names=names)
            score = m.score(**args(val_set))
        cands.append((th, score))
    best = max(c[1] for c in cands)
    th = max(c[0] for c in cands if c[1] >= best - 0.02)
    m = make_model(th, degree)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(**args(tr), feature_names=names)
    return m, th, args


def _blowup(t, x):
    return 20.0 - np.max(np.abs(x))  # data is scaled to |x| <= 1


_blowup.terminal = True


def rollout(model, beat, ph, dt, forced):
    t = np.arange(len(beat)) * dt
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sim = model.simulate(beat[0], t, u=phase_input(ph) if forced else None,
                                 integrator_kws={'method': 'LSODA', 'rtol': 1e-6, 'atol': 1e-8,
                                                 'events': _blowup})
    except Exception:
        return None, DIVERGED
    # With an array input pysindy integrates over t[:-1], so a full forced
    # rollout is one sample short. Anything shorter stopped early (blow-up
    # event or integrator failure).
    if len(sim) < len(beat) - 1 or not np.all(np.isfinite(sim)):
        return None, DIVERGED
    beat = beat[:len(sim)]
    nrmse = np.sqrt(np.mean((sim - beat) ** 2)) / np.std(beat)
    return sim, float(min(nrmse, DIVERGED))


def evaluate(rep, dt):
    res = {}
    for kind, (degree, forced) in MODELS.items():
        model, th, args = fit(rep, degree, forced, dt)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r2 = float(model.score(**args(rep['test'])))
        sims, errs = [], []
        for beat, ph in rep['test']:
            sim, e = rollout(model, beat, ph, dt, forced)
            sims.append(sim)
            errs.append(e)
        res[kind] = dict(
            threshold=th, r2_deriv=r2, n_terms=int(np.count_nonzero(model.coefficients())),
            rollout_nrmse_median=float(np.median(errs)),
            diverged=int(sum(e >= DIVERGED for e in errs)), n_test=len(errs),
            equations=model.equations(precision=2), sims=sims)
    return res


def phase_line(ax, xy, ph, **kw):
    pts = xy[:, None, :2]
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap='twilight_shifted',
                        norm=plt.Normalize(0, 1), **kw)
    lc.set_array(ph[:-1])
    ax.add_collection(lc)


if __name__ == '__main__':
    x, fs = ms.load_ecg('focus_data/007c')
    peaks = ms.find_r_peaks(x, fs)
    reps = {k: normalize(v) for k, v in build_representations(x, fs, peaks).items()}
    dt = float(1.0 / (fs * np.median(np.diff(peaks) / fs)))  # time step in beats

    results = {}
    for name, rep in reps.items():
        print(f'fitting {name} ({len(rep["train"])} train / {len(rep["test"])} test beats)', flush=True)
        results[name] = evaluate(rep, dt)
        for kind in MODELS:
            r = results[name][kind]
            print(f'{name:18s} {kind:16s} th={r["threshold"]:<5} R2={r["r2_deriv"]:.3f} '
                  f'terms={r["n_terms"]:3d} rollout NRMSE={r["rollout_nrmse_median"]:.3f} '
                  f'diverged={r["diverged"]}/{r["n_test"]}', flush=True)

    # Figure: rows = representation; cols = model (x-y plane)
    n_show = 3
    fig, axs = plt.subplots(len(reps), len(MODELS), figsize=(5 * len(MODELS), 3.2 * len(reps)))
    for i, (name, rep) in enumerate(reps.items()):
        for j, kind in enumerate(MODELS):
            ax, r = axs[i, j], results[name][kind]
            for (beat, ph), sim in list(zip(rep['test'], r['sims']))[:n_show]:
                phase_line(ax, beat, ph, linewidths=2.5, alpha=0.35)
                if sim is not None:
                    ax.plot(sim[:, 0], sim[:, 1], color='k', lw=0.9)
            ax.autoscale()
            ax.set_title(f'{name}: {kind}\nR²={r["r2_deriv"]:.2f}  terms={r["n_terms"]}  '
                         f'NRMSE={r["rollout_nrmse_median"]:.2f}  diverged {r["diverged"]}/{r["n_test"]}',
                         fontsize=9, loc='left')
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SINDy per representation: test beats (colour = cardiac phase) vs model rollout (black)',
                 fontsize=11)
    fig.tight_layout()
    plt.savefig('imgs/sindy_multiscale.png', dpi=100)

    for name in results:
        for kind in results[name]:
            results[name][kind].pop('sims')
    with open('sindy_results.json', 'w') as f:
        json.dump(dict(train_s=TRAIN, test_s=TEST, dt_beats=dt, results=results), f, indent=1)
