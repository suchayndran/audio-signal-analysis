#!/usr/bin/env python3
"""
Audio Signal Analysis and Visualization
Signals and Systems course project, IIITDM Kancheepuram

What this script does
  1. Uses a WAV file you give it, or builds a test signal (3 notes + a 4 kHz whine + noise)
  2. Plots the time-domain waveform
  3. Computes the FFT spectrum (rectangular vs Hann window) and lists the strongest peaks
  4. Applies a Butterworth low-pass filter and plots spectrograms before and after
  5. Computes MFCCs step by step (frame -> FFT -> mel filters -> log -> DCT)
  6. Saves every plot, two WAV files and a results.txt summary in the "outputs" folder

It needs only numpy, scipy and matplotlib, so it runs the same on your laptop,
in Google Colab and on GitHub (see .github/workflows/run-analysis.yml).

Examples
  python audio_analysis.py                      # test signal, no file needed
  python audio_analysis.py --input my_clip.wav  # your own WAV file
  python audio_analysis.py --cutoff 3000        # change the filter cutoff
"""

import argparse
import os
import sys

import matplotlib

if "--show" not in sys.argv:        # no window needed (servers, GitHub Actions)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.io import wavfile

BLUE, LIGHT_BLUE, BROWN, GREY, GRID = "#214B8C", "#8FA9D3", "#8C5A1E", "#55606E", "#D5DBE6"
TEST_NOTES = (440.0, 554.37, 659.25)     # A4, C#5, E5
WHINE_HZ = 4000.0


# ----------------------------------------------------------------------------
# 1. Getting a signal
# ----------------------------------------------------------------------------
def make_test_signal(sr=22050, seconds=3.0, seed=0):
    """Three notes (each with a 2nd harmonic), a 4 kHz whine and random noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for i, f in enumerate(TEST_NOTES):
        in_note = (t >= i) & (t < i + 1)
        tt = t[in_note] - i
        tone = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)
        x[in_note] = tone * np.exp(-2 * tt)          # each note fades out
    x += 0.15 * np.sin(2 * np.pi * WHINE_HZ * t)     # the unwanted whine
    x += 0.05 * rng.standard_normal(len(t))          # background noise
    return x.astype(np.float32), sr


def read_wav(path, max_seconds):
    """Read a WAV file as mono floats between -1 and 1."""
    sr, data = wavfile.read(path)
    if data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)                      # stereo -> mono
    data = data[: int(max_seconds * sr)]
    return data.astype(np.float32), sr


# ----------------------------------------------------------------------------
# 2. Frequency analysis
# ----------------------------------------------------------------------------
def fft_spectrum(x, sr, window=True):
    """Single-sided amplitude spectrum. A Hann window cuts spectral leakage."""
    w = np.hanning(len(x)) if window else np.ones(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / w.sum() * 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return freqs, mag


def to_db(a):
    return 20 * np.log10(np.maximum(a, 1e-12))


def strongest_peaks(freqs, mag, n=8, within_db=35, min_gap_hz=15):
    """The n strongest spectral peaks, within `within_db` of the highest one."""
    db = to_db(mag)
    df = freqs[1] - freqs[0]
    idx, _ = signal.find_peaks(db, height=db.max() - within_db, prominence=10,
                               distance=max(1, int(min_gap_hz / df)))
    idx = idx[np.argsort(db[idx])[::-1]][:n]
    return [(float(freqs[i]), float(db[i])) for i in idx]


# ----------------------------------------------------------------------------
# 3. Filtering
# ----------------------------------------------------------------------------
def lowpass(x, sr, cutoff, order=5):
    """Butterworth low-pass, applied forward and backward so nothing shifts in time."""
    sos = signal.butter(order, cutoff, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


# ----------------------------------------------------------------------------
# 4. STFT and MFCC (written out step by step)
# ----------------------------------------------------------------------------
def stft_magnitude(x, n_fft=2048, hop=512):
    """Short-time Fourier transform magnitude, shape (n_fft/2 + 1, frames)."""
    window = signal.get_window("hann", n_fft, fftbins=True)
    padded = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    idx = np.arange(n_fft)[:, None] + hop * np.arange(n_frames)[None, :]
    return np.abs(np.fft.rfft(padded[idx] * window[:, None], axis=0))


def hz_to_mel(f):
    f = np.asarray(f, dtype=float)
    step = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / step, f / (200.0 / 3.0))


def mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    step = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(step * (m - 15.0)), m * (200.0 / 3.0))


def mel_filterbank(sr, n_fft=2048, n_mels=40):
    """Triangular filters that are evenly spaced on the mel scale."""
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
    gaps = np.diff(edges)
    ramps = edges[:, None] - fft_freqs[None, :]
    rising = -ramps[:-2] / gaps[:-1, None]
    falling = ramps[2:] / gaps[1:, None]
    bank = np.maximum(0, np.minimum(rising, falling))
    return bank * (2.0 / (edges[2:] - edges[:-2]))[:, None]


def mfcc(x, sr, n_mfcc=13, n_fft=2048, hop=512, n_mels=40):
    """frame + window -> FFT -> mel filters -> log -> DCT -> keep first n_mfcc."""
    power = stft_magnitude(x, n_fft, hop) ** 2
    mel_energy = mel_filterbank(sr, n_fft, n_mels) @ power
    log_mel = 10 * np.log10(np.maximum(mel_energy, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 80)       # ignore anything 80 dB below the peak
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=BLUE, fontweight="bold")
    ax.set_xlabel(xlabel, color=GREY)
    ax.set_ylabel(ylabel, color=GREY)
    ax.tick_params(colors=GREY)
    ax.grid(color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=170)
    return fig


def plot_waveform(x, y, sr, cutoff, out_dir):
    t = np.arange(len(x)) / sr
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, x, color=LIGHT_BLUE, linewidth=0.5, label="Original")
    ax.plot(t, y, color=BLUE, linewidth=0.6, label=f"After {cutoff:.0f} Hz low-pass")
    ax.set_xlim(0, t[-1])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize="small")
    style(ax, "Time-domain waveform", "Time (s)", "Amplitude")
    return save(fig, out_dir, "01_waveform.png")


def plot_spectrum(freqs, rect, hann, peaks, sr, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(freqs, to_db(rect), color=LIGHT_BLUE, linewidth=0.6, label="Rectangular (no window)")
    ax.plot(freqs, to_db(hann), color=BROWN, linewidth=0.7, label="Hann window")
    if peaks:
        f0, d0 = peaks[0]
        ax.axvline(f0, color=BLUE, linestyle="--", linewidth=0.8)
        ax.annotate(f"strongest peak: {f0:.0f} Hz", (f0, d0), xytext=(f0 + 120, d0 - 4), color=BLUE, fontsize="small")
    ax.set_xlim(0, min(6000, sr / 2))
    ax.set_ylim(-100, 0)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    style(ax, "FFT magnitude spectrum", "Frequency (Hz)", "Magnitude (dB)")
    return save(fig, out_dir, "02_fft_spectrum.png")


def plot_spectrograms(mag_x, mag_y, sr, hop, cutoff, out_dir):
    ref = mag_x.max()
    duration = (mag_x.shape[1] - 1) * hop / sr
    top = min(6000, sr / 2)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, mag, name in zip(axes, (mag_x, mag_y), ("Original", f"After {cutoff:.0f} Hz low-pass filter")):
        last = ax is axes[-1]
        db = np.maximum(to_db(mag / ref), -80)
        im = ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                       extent=[0, duration, 0, sr / 2])
        ax.set_ylim(0, top)
        style(ax, name, "Time (s)" if last else "", "Frequency (Hz)")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.04)
    cbar.set_label("dB", color=GREY)
    cbar.set_ticks([-80, -60, -40, -20, 0])
    fig.savefig(os.path.join(out_dir, "03_spectrogram.png"), dpi=170, bbox_inches="tight")
    return fig


def plot_mfcc(coeffs, duration, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(coeffs, origin="lower", aspect="auto", cmap="coolwarm", extent=[0, duration, 0.5, coeffs.shape[0] + 0.5])
    ax.set_yticks(range(1, coeffs.shape[0] + 1, 2))
    fig.colorbar(im, ax=ax, pad=0.015).set_label("value", color=GREY)
    style(ax, f"MFCC ({coeffs.shape[0]} coefficients)", "Time (s)", "MFCC number")
    ax.grid(False)
    return save(fig, out_dir, "04_mfcc.png")


def write_wav(path, x, sr):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Audio signal analysis and visualization")
    ap.add_argument("--input", help="path to a WAV file (default: built-in test signal)")
    ap.add_argument("--cutoff", type=float, default=1500, help="low-pass cutoff in Hz (default 1500)")
    ap.add_argument("--order", type=int, default=5, help="Butterworth filter order (default 5)")
    ap.add_argument("--n-fft", type=int, default=2048, help="STFT window length in samples (default 2048)")
    ap.add_argument("--hop", type=int, default=512, help="STFT hop in samples (default 512)")
    ap.add_argument("--max-seconds", type=float, default=10, help="only use the first N seconds of a file")
    ap.add_argument("--out", default="outputs", help="folder for results (default: outputs)")
    ap.add_argument("--show", action="store_true", help="also open the plot windows")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- signal
    using_test = args.input is None
    x, sr = make_test_signal() if using_test else read_wav(args.input, args.max_seconds)
    if len(x) <= args.n_fft:
        sys.exit(f"The audio is too short ({len(x)} samples). It needs more than {args.n_fft} samples.")
    cutoff = min(args.cutoff, 0.45 * sr)
    duration = len(x) / sr

    # --- analysis
    y = lowpass(x, sr, cutoff, args.order)
    freqs, rect = fft_spectrum(x, sr, window=False)
    _, hann = fft_spectrum(x, sr, window=True)
    _, hann_after = fft_spectrum(y, sr, window=True)
    peaks = strongest_peaks(freqs, hann)
    mag_x = stft_magnitude(x, args.n_fft, args.hop)
    mag_y = stft_magnitude(y, args.n_fft, args.hop)
    coeffs = mfcc(y, sr, n_fft=args.n_fft, hop=args.hop)
    above = freqs > cutoff
    energy_drop = 10 * np.log10(np.sum(hann[above] ** 2) / max(np.sum(hann_after[above] ** 2), 1e-30))

    # --- outputs
    plot_waveform(x, y, sr, cutoff, args.out)
    plot_spectrum(freqs, rect, hann, peaks, sr, args.out)
    plot_spectrograms(mag_x, mag_y, sr, args.hop, cutoff, args.out)
    plot_mfcc(coeffs, duration, args.out)
    write_wav(os.path.join(args.out, "filtered.wav"), y, sr)
    if using_test:
        write_wav(os.path.join(args.out, "test_signal.wav"), x, sr)

    lines = [
        "AUDIO SIGNAL ANALYSIS - RESULTS",
        "-------------------------------",
        f"Signal            : {'built-in test signal' if using_test else args.input}",
        f"Sampling rate     : {sr} Hz",
        f"Duration          : {duration:.2f} s ({len(x)} samples)",
        f"Nyquist frequency : {sr / 2:.0f} Hz",
        f"Filter            : Butterworth low-pass, order {args.order}, cutoff {cutoff:.0f} Hz",
        f"STFT              : window {args.n_fft} samples ({1000 * args.n_fft / sr:.0f} ms), "
        f"hop {args.hop} samples, frequency step {sr / args.n_fft:.1f} Hz",
        f"Energy above cutoff reduced by {energy_drop:.0f} dB",
        f"MFCC shape        : {coeffs.shape[0]} coefficients x {coeffs.shape[1]} frames",
        "",
        "Strongest spectral peaks (Hann window):",
    ]
    lines += [f"  {f:8.1f} Hz   {d:6.1f} dB" for f, d in peaks]

    # --- self-check on the test signal, so a run proves the code works
    ok = True
    if using_test:
        lines += ["", "Self-check on the test signal:"]
        found = [f for f, _ in peaks]
        for target in TEST_NOTES + (WHINE_HZ,):
            hit = any(abs(f - target) < 3 for f in found)
            ok &= hit
            lines.append(f"  peak near {target:7.2f} Hz found : {'PASS' if hit else 'FAIL'}")
        band = (freqs > WHINE_HZ - 20) & (freqs < WHINE_HZ + 20)
        drop = to_db(hann[band].max()) - to_db(hann_after[band].max())
        ok &= drop > 60
        lines.append(f"  4 kHz whine reduced by {drop:.0f} dB (needs > 60) : {'PASS' if drop > 60 else 'FAIL'}")
        lines.append("  RESULT: " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nPlots and results saved in the '{args.out}' folder.")

    if args.show:
        plt.show()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
