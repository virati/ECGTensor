"""Export the multiscale phase-portrait data used by the ECG Phase Atlas.

Usage: python export_atlas.py [out.json]
"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import multiscale as ms
from sklearn.decomposition import PCA

x, fs = ms.load_ecg('focus_data/007c')
pk = ms.find_r_peaks(x, fs)
T0, T1 = 30.0, 42.0
a, b = int(T0 * fs), int(T1 * fs)
seg_pk = pk[(pk >= a) & (pk < b)]
rr = float(np.median(np.diff(pk)) / fs)


def norm(arrs):
    allp = np.concatenate(arrs)
    c = np.median(allp, axis=0)
    s = np.abs(allp - c).max()
    return [(q - c) / s for q in arrs]


def split_cont(y):
    beats, t0s = [], []
    for p, q in zip(seg_pk[:-1], seg_pk[1:]):
        seg = y[p:q + 1, :3]
        ph = np.arange(q - p + 1) / (q - p)
        beats.append(np.column_stack([seg, np.minimum(ph, 0.999)]))
        t0s.append(p / fs)
    return beats, t0s


def pack(name, short, axes, beats, t0s, blurb):
    xyz = norm([bb[:, :3] for bb in beats])
    out = [np.column_stack([q, bb[:, 3]]).round(4).tolist() for q, bb in zip(xyz, beats)]
    return dict(name=name, short=short, axes=axes, beats=out,
                t0=[round(t, 4) for t in t0s], blurb=blurb)


methods = []
methods.append(pack('Raw', 'raw', ['V1', 'V2', 'V3'], *split_cont(x),
                    'Band-passed 0.5–40 Hz, 250 Hz. QRS loop dominates; P and T crowd the origin.'))
xl = ms.signed_log(x)
methods.append(pack('Signed log₁₀', 'slog', ['slog V1', 'slog V2', 'slog V3'], *split_cont(xl),
                    'sign(x)·log₁₀(1+|x|/x₀), x₀ = per-lead median |x|. Compresses QRS, opens the T loop.'))

beats, bp = ms.beat_stack(x, fs, pk)
wb = ms.wavelet2d(beats)
tb = np.linspace(-0.25, 0.55, beats.shape[1])
phb = np.mod(tb / rr, 1.0)
sel = np.where((bp >= a) & (bp < seg_pk[-1]))[0]
methods.append(pack('2D wavelet', 'wav', ['V1', 'V2', 'V3'],
                    [np.column_stack([wb[i, :, :3], phb]) for i in sel],
                    [bp[i] / fs - 0.25 for i in sel],
                    'Beats stacked into a beat×time image, undecimated 2D wavelet (5 levels): cross-beat denoise, fine scales attenuated.'))

alpha = -0.3
xf = ms.gl_fractional_derivative(x, alpha, fs=fs)
methods.append(pack('Fractional D⁻⁰·³', 'frac', ['D⁻⁰·³V1', 'D⁻⁰·³V2', 'D⁻⁰·³V3'], *split_cont(xf),
                    'Grünwald–Letnikov fractional integral. Power-law kernel, no preferred time scale; tilts the spectrum toward slow P/T.'))

z, rz = ms.matryoshka_embed(x, return_recon=True)
methods.append(pack('Matryoshka', 'mrl', ['z0', 'z1', 'z2'], *split_cont(z),
                    'Autoencoder on 128 ms windows of all 9 leads; every prefix z[:m] must reconstruct. z[:3] shown.'))
zl, rzl = ms.matryoshka_embed(xl, return_recon=True)
methods.append(pack('Log + Matryoshka', 'slogmrl', ['z0', 'z1', 'z2'], *split_cont(zl),
                    'Matryoshka trained on signed-log leads, so the loss is not all QRS. Separate QRS, T and P loops.'))


def pca_resid(y):
    W = ms._windows(y, 32)
    W = (W - W.mean(0)) / (W.std(0) + 1e-8)
    p = PCA(16).fit(W)
    return (1 - np.cumsum(p.explained_variance_ratio_)).tolist()


recon = dict(mrl=rz.tolist(), mrl_slog=rzl.tolist(),
             pca=pca_resid(x), pca_slog=pca_resid(xl))

json.dump(dict(fs=fs, rr=rr, t0=T0, record='STAFF-III 007c', methods=methods, recon=recon),
          open(sys.argv[1] if len(sys.argv) > 1 else 'web/public/ecg_data.json', 'w'), separators=(',', ':'))
print('beats', [len(m['beats']) for m in methods])
print({k: np.round(v[:4], 3).tolist() for k, v in recon.items()})
