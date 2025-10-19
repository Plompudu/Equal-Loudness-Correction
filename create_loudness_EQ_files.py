"""
equal_loudness_batch.py
Generate equal-loudness-tonality correction PEQs for a range of listening levels.

- Uses ISO226 equal-loudness contours
- Computes EQ to make given listening level sound like the reference
- Optimizes REW-style parametric EQs (gain & Q)
- Creates folders:
    ./PEQ_85dB_reference/
    ./PEQ_84dB_reference/
    ...
  containing individual .txt EQ filter sets for each listening level.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import os
import math
from numba import njit

# ============================================================
# 1) ISO226 data
# ============================================================
iso_freq = np.array([
    20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,
    800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,
    12500,16000,20000
])
raw_contours = {
    0:   np.array([76.55,65.62,55.12,45.53,37.63,30.86,25.02,20.51,16.65,13.12,10.09,7.54,5.11,3.06,1.48,0.3,-0.3,-0.01,1.03,-1.19,-4.11,-7.05,-9.03,-8.49,-4.48,3.28,9.83,10.48,8.38,14.1,79.65]),
    10:  np.array([83.75,75.76,68.21,61.14,54.96,49.01,43.24,38.13,33.48,28.77,24.84,21.33,18.05,15.14,12.98,11.18,9.99,10.0,11.26,10.43,7.27,4.45,3.04,3.8,7.46,14.35,20.98,23.43,22.33,25.17,81.47]),
    20:  np.array([89.58,82.65,75.98,69.62,64.02,58.55,53.19,48.38,43.94,39.37,35.51,31.99,28.69,25.67,23.43,21.48,20.1,20.01,21.46,21.4,18.15,15.38,14.26,15.14,18.63,25.02,31.52,34.43,33.04,34.67,84.18]),
    40:  np.array([99.85,93.94,88.17,82.63,77.78,73.08,68.48,64.37,60.59,56.7,53.41,50.4,47.58,44.98,43.05,41.34,40.06,40.01,41.82,42.51,39.23,36.51,35.61,36.65,40.01,45.83,51.8,54.28,51.49,51.96,92.77]),
    60:  np.array([109.51,104.23,99.08,94.18,89.96,85.94,82.05,78.65,75.56,72.47,69.86,67.53,65.39,63.45,62.05,60.81,59.89,60.01,62.15,63.19,59.96,57.26,56.42,57.57,60.89,66.36,71.66,73.16,68.63,68.43,104.92]),
    80:  np.array([118.99,114.23,109.65,105.34,101.72,98.36,95.17,92.48,90.09,87.82,85.92,84.31,82.89,81.68,80.86,80.17,79.67,80.01,82.48,83.74,80.59,77.88,77.07,78.31,81.62,86.81,91.41,91.74,85.41,84.67,118.95]),
    100: np.array([128.41,124.15,120.11,116.38,113.35,110.65,108.16,106.17,104.48,103.03,101.85,100.97,100.3,99.83,99.62,99.5,99.44,100.01,102.81,104.25,101.18,98.48,97.67,99.0,102.3,107.23,111.11,110.23,102.07,100.83,133.73]),
}

phon_levels = np.array(sorted(raw_contours.keys()))
spl_matrix = np.vstack([raw_contours[p] for p in phon_levels])
interp_funcs = [interp1d(phon_levels, spl_matrix[:, i], kind="cubic", bounds_error=False, fill_value="extrapolate") for i in range(len(iso_freq))]

def spl_at(phon): return np.array([f(phon) for f in interp_funcs])

# ============================================================
# 2) Target EQ response
# ============================================================
def make_target_response(listening_phon, reference_phon, n_fft=8192, fmin=20, fmax=20000):
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_fft)
    spl_ref = spl_at(reference_phon)
    spl_listen = spl_at(listening_phon)
    gain_db_at_iso = spl_listen - spl_ref  # correct polarity
    gain_db_dense = np.interp(freqs, iso_freq, gain_db_at_iso)
    return freqs, gain_db_dense

# ============================================================
# 3) Filter utilities
# ============================================================
@njit
def peaking_biquad_numba(fs, fc, Q, gain_db):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    alpha = math.sin(w0) / (2.0 * Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

@njit
def freq_response_numba(b, a, fs, freqs):
    H = np.empty(freqs.shape, dtype=np.complex128)
    for i in range(freqs.size):
        w = 2.0 * math.pi * freqs[i] / fs
        ejw = complex(math.cos(-w), math.sin(-w))
        ejw2 = ejw * ejw
        H[i] = (b[0] + b[1]*ejw + b[2]*ejw2) / (a[0] + a[1]*ejw + a[2]*ejw2)
    return H

@njit
def combined_H_numba(bands_fc, bands_Q, bands_gain, fs, freqs):
    H = np.ones(freqs.shape, dtype=np.complex128)
    n_bands = bands_fc.size
    for j in range(n_bands):
        b, a = peaking_biquad_numba(fs, bands_fc[j], bands_Q[j], bands_gain[j])
        H *= freq_response_numba(b, a, fs, freqs)
    return H

def build_initial_filters(n_filters, freqs, target_db):
    edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_filters+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    bands = []
    for i in range(n_filters):
        mask = (freqs >= edges[i]) & (freqs < edges[i+1])
        gain = np.mean(target_db[mask]) if np.any(mask) else 0.0
        B = math.log2(edges[i+1]/edges[i])
        Q = 1.0 / (2**(B/2) - 2**(-B/2))
        bands.append({'fc': centers[i], 'gain_db': gain, 'Q': Q})
    return bands

def optimize_bands(bands, fs, freqs, target_db):
    x0 = []
    for b in bands:
        x0 += [b['gain_db'], np.log10(b['Q'])]
    x0 = np.array(x0)

    bands_fc = np.array([b['fc'] for b in bands])
    bands_Q  = np.array([b['Q'] for b in bands])
    bands_gain = np.array([b['gain_db'] for b in bands])

    def residuals(x):
        for i in range(len(bands)):
            bands_gain[i] = x[2*i]
            bands_Q[i] = 10 ** x[2*i + 1]
        H = combined_H_numba(bands_fc, bands_Q, bands_gain, fs, freqs)
        out_db = 20*np.log10(np.maximum(np.abs(H), 1e-12))
        return out_db - target_db

    res = least_squares(residuals, x0, verbose=0, max_nfev=200)
    for i, b in enumerate(bands):
        b['gain_db'] = res.x[2*i]
        b['Q'] = 10 ** res.x[2*i + 1]

    rms = np.sqrt(np.mean(res.fun**2))
    return bands, rms

def rew_lines(bands):
    return [f"Filter {i}: ON PK Fc {b['fc']:.1f} Hz Gain {b['gain_db']:+.2f} dB Q {b['Q']:.2f}"
            for i,b in enumerate(bands,1)]

# ============================================================
# 4) Generate one EQ file
# ============================================================
def generate_eq_file(listening_phon, reference_phon, out_path, overwrite=False, fs=48000, n_fft=8192, n_filters=10):
    """Generate one EQ file or a blank placeholder if same SPLs."""
    if os.path.exists(out_path) and not overwrite:
        print(f"⏭️ Skipped (exists): {os.path.basename(out_path)}")
        return

    if listening_phon == reference_phon:
        # Create a blank file with a simple comment header
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Listening {listening_phon} => Reference {reference_phon}\n")
        print(f"🟢 Created blank placeholder: {os.path.basename(out_path)}")
        return

    freqs, target_db = make_target_response(listening_phon, reference_phon, n_fft=n_fft)
    bands = build_initial_filters(n_filters, freqs, target_db)
    bands_opt, rms_opt = optimize_bands(bands, fs, freqs, target_db)
    lines = rew_lines(bands_opt)
    header = f"# Listening {listening_phon} => Reference {reference_phon}\n# RMS error: {rms_opt:.3f} dB"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join([header] + lines))
    print(f"✅ {os.path.basename(out_path)} | RMS: {rms_opt:.3f} dB")

# ============================================================
# 5) Batch generation
# ============================================================
def batch_generate(sets, overwrite=False):
    fs = 48000
    n_fft = 8192
    n_filters = 10

    for s in sets:
        ref = s["ref"]
        os.makedirs(s["folder"], exist_ok=True)
        print(f"\n=== Generating {s['folder']} (reference {ref} dB) ===")
        for lvl in s["range"]:
            out_file = os.path.join(s["folder"], f"{lvl}dB_to_{ref}dB.txt")
            generate_eq_file(lvl, ref, out_file, overwrite=overwrite, fs=fs, n_fft=n_fft, n_filters=n_filters)
        print(f"✅ Finished {s['folder']}")

# ============================================================
# 6) Run
# ============================================================
if __name__ == "__main__":
    # Toggle overwrite behavior here
    OVERWRITE_FILES = False  # ⬅️ set to True to overwrite existing files

    sets = [
        {"ref": 85, "range": range(30, 85 + 1), "folder": "PEQ_85dB_reference"},
        {"ref": 84, "range": range(30, 85 + 1), "folder": "PEQ_84dB_reference"},
        {"ref": 83, "range": range(30, 85 + 1), "folder": "PEQ_83dB_reference"},
        {"ref": 82, "range": range(30, 85 + 1), "folder": "PEQ_82dB_reference"},
        {"ref": 81, "range": range(30, 85 + 1), "folder": "PEQ_81dB_reference"},
        {"ref": 80, "range": range(30, 85 + 1), "folder": "PEQ_80dB_reference"},
        {"ref": 79, "range": range(30, 85 + 1), "folder": "PEQ_79dB_reference"},
        {"ref": 78, "range": range(30, 85 + 1), "folder": "PEQ_78dB_reference"},
        {"ref": 77, "range": range(30, 85 + 1), "folder": "PEQ_77dB_reference"},
        {"ref": 76, "range": range(30, 85 + 1), "folder": "PEQ_76dB_reference"},
        {"ref": 75, "range": range(30, 85 + 1), "folder": "PEQ_75dB_reference"},
        {"ref": 74, "range": range(30, 85 + 1), "folder": "PEQ_74dB_reference"},
        {"ref": 73, "range": range(30, 85 + 1), "folder": "PEQ_73dB_reference"},
        {"ref": 72, "range": range(30, 85 + 1), "folder": "PEQ_72dB_reference"},
        {"ref": 71, "range": range(30, 85 + 1), "folder": "PEQ_71dB_reference"},
        {"ref": 70, "range": range(30, 85 + 1), "folder": "PEQ_70dB_reference"},
        {"ref": 69, "range": range(30, 85 + 1), "folder": "PEQ_69dB_reference"},
        {"ref": 68, "range": range(30, 85 + 1), "folder": "PEQ_68dB_reference"},
        {"ref": 67, "range": range(30, 85 + 1), "folder": "PEQ_67dB_reference"},
        {"ref": 66, "range": range(30, 85 + 1), "folder": "PEQ_66dB_reference"},
        {"ref": 65, "range": range(30, 85 + 1), "folder": "PEQ_65dB_reference"},
    ]

    batch_generate(sets, overwrite=OVERWRITE_FILES)
