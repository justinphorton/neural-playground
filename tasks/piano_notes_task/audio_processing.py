"""Read piano WAVs and prepare the same waveform for every experiment trial."""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import hashlib

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .stimulus import apply_fade_out


NOTES = ("C4", "D4", "E4")
STIMULUS_RATE = 10_000  # 0.1 ms per sample, matching the planned livn stimulus step
NOTE_SECONDS = 1.0  # one fixed stimulus window following the detected strike


@dataclass(frozen=True)
class NoteSample:
    path: Path
    note: str


@dataclass(frozen=True)
class PreparedAudio:
    samples: np.ndarray  # mono, signed, peak-normalized; short clips padded with zeros
    sample_rate: int
    original_duration_s: float
    start_s: float
    normalization_peak: float = 1.0
    active_end_s: float | None = None

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class AudioAudit:
    path: str
    file_hash: str
    original_sample_rate: int
    original_channels: int
    original_duration_s: float
    detected_onset_s: float
    extracted_start_s: float
    extracted_end_s: float
    original_peak: float
    original_rms: float
    original_dc_offset: float
    corrected_dc_offset: float
    normalized_peak: float
    normalized_rms: float
    current_peak_ua: float
    current_rms_ua: float
    final_5ms_rms_ua: float
    final_10ms_rms_ua: float
    stimulus_hash: str


def discover_samples(root: Path, split: str) -> list[NoteSample]:
    """Find WAVs by note folder each time the training or test run starts."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {split_dir}")
    samples = [
        NoteSample(path, note)
        for note in NOTES
        for path in sorted((split_dir / note).glob("*"))
        if path.is_file() and path.suffix.lower() == ".wav"
    ]
    return samples


def prepare_audio(path: Path, stimulus_rate: int = STIMULUS_RATE) -> PreparedAudio:
    """Downmix, take the first second of a strike, resample, and peak-normalize."""
    if stimulus_rate <= 0:
        raise ValueError("stimulus_rate must be positive")
    sample_rate, recording = wavfile.read(path)
    if recording.size == 0:
        raise ValueError(f"Empty WAV: {path}")
    if np.issubdtype(recording.dtype, np.integer):
        if recording.dtype == np.uint8:
            waveform = (recording.astype(np.float64) - 128) / 128
        else:
            waveform = recording.astype(np.float64) / max(abs(np.iinfo(recording.dtype).min), 1)
    elif np.issubdtype(recording.dtype, np.floating):
        waveform = recording.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV sample format: {recording.dtype}")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Expected mono or stereo WAV: {path}")
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Non-finite audio values: {path}")
    waveform -= waveform.mean()  # remove a constant recording offset

    frame_size = max(1, round(sample_rate * 0.01))
    frame_count = len(waveform) // frame_size
    if frame_count == 0:
        raise ValueError(f"WAV is too short to find a note: {path}")
    frames = waveform[: frame_count * frame_size].reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(frame_rms.max())
    if peak < 1e-7:
        raise ValueError(f"WAV is silent: {path}")
    noise_floor = float(np.percentile(frame_rms, 10))
    # A short recording may contain only the note, so its 10th percentile can
    # be nearly as loud as its peak. Keep the threshold below the peak then.
    threshold = min(max(4 * noise_floor, 0.01 * peak), 0.25 * peak)
    active = np.flatnonzero(frame_rms > threshold)
    if len(active) == 0:
        raise ValueError(f"No active note found in: {path}")

    start = max(0, int(active[0] * frame_size - 0.02 * sample_rate))
    stop = min(len(waveform), start + round(NOTE_SECONDS * sample_rate))
    selected = waveform[start:stop]
    factor = gcd(sample_rate, stimulus_rate)
    converted = resample_poly(selected, stimulus_rate // factor, sample_rate // factor)
    absolute_peak = float(np.max(np.abs(converted)))
    if absolute_peak < 1e-7:
        raise ValueError(f"Detected note is silent: {path}")
    normalized = (converted / absolute_peak).astype(np.float32)
    target_steps = round(NOTE_SECONDS * stimulus_rate)
    normalized = np.pad(normalized[:target_steps], (0, max(0, target_steps - len(normalized))))
    return PreparedAudio(
        normalized,
        stimulus_rate,
        len(waveform) / sample_rate,
        start / sample_rate,
        normalization_peak=absolute_peak,
        active_end_s=min(len(waveform) / sample_rate, (start + round(NOTE_SECONDS * sample_rate)) / sample_rate),
    )


def audit_audio(path: Path, peak_current_ua: float = 25.0, stimulus_rate: int = STIMULUS_RATE, fade_out_ms: float = 15.0) -> AudioAudit:
    """Return auditable source, preprocessing, and current statistics."""
    path = Path(path)
    sample_rate, recording = wavfile.read(path)
    channels = 1 if recording.ndim == 1 else recording.shape[1]
    if np.issubdtype(recording.dtype, np.integer):
        scale = 128 if recording.dtype == np.uint8 else max(abs(np.iinfo(recording.dtype).min), 1)
        original = (recording.astype(np.float64) - (128 if recording.dtype == np.uint8 else 0)) / scale
    else:
        original = recording.astype(np.float64)
    mono = original.mean(axis=1) if original.ndim == 2 else original
    original_dc = float(np.mean(mono))
    corrected = mono - original_dc
    original_peak = float(np.max(np.abs(mono)))
    original_rms = float(np.sqrt(np.mean(mono * mono)))
    prepared = prepare_audio(path, stimulus_rate=stimulus_rate)
    tapered = apply_fade_out(prepared.samples, stimulus_rate, fade_out_ms)
    current = tapered * float(peak_current_ua)
    digest = hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
    return AudioAudit(
        path=str(path.resolve()), file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_sample_rate=int(sample_rate), original_channels=int(channels),
        original_duration_s=float(len(mono) / sample_rate), detected_onset_s=float(prepared.start_s + 0.02),
        extracted_start_s=float(prepared.start_s), extracted_end_s=float(prepared.active_end_s or prepared.start_s + 1.0),
        original_peak=original_peak, original_rms=original_rms, original_dc_offset=original_dc,
        corrected_dc_offset=float(np.mean(corrected)), normalized_peak=float(np.max(np.abs(prepared.samples))),
        normalized_rms=float(np.sqrt(np.mean(prepared.samples * prepared.samples))),
        current_peak_ua=float(np.max(np.abs(current))), current_rms_ua=float(np.sqrt(np.mean(current * current))),
        final_5ms_rms_ua=float(np.sqrt(np.mean(current[-round(stimulus_rate * 0.005):] ** 2))),
        final_10ms_rms_ua=float(np.sqrt(np.mean(current[-round(stimulus_rate * 0.010):] ** 2))), stimulus_hash=digest,
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    samples = discover_samples(root, "training_data")
    if not samples:
        raise SystemExit("No training WAV files found in C4, D4, or E4 folders")
    for sample in samples:
        audio = prepare_audio(sample.path)
        print(f"{sample.note}: {sample.path.name}: "
              f"{audio.original_duration_s:.2f}s file -> "
              f"{audio.duration_s:.2f}s note, "
              f"{audio.sample_rate} Hz, peak 1.0")


if __name__ == "__main__":
    main()
