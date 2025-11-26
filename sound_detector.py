import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import tempfile
import librosa
import numpy as np
from torchvision.models import resnet18
import uuid
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.signal as signal
import math

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
FIXED_FRAMES = 256

FRAME_WINDOW_FRAMES = 64
FRAME_HOP_FRAMES = 16
FRAME_CONF_THRESH = 0.6
MERGE_GAP_FRAMES = 2
PLOT_OUTPUT_DIR = "plots"
PLOT_OUTPUT_DIR = os.path.join('static', 'plots')
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

CHUNK_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.65

def build_model(num_classes):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class_names = {
    0: "SpeedBoat",
    1: "UUV",
    2: "KaiYuan",
}

def load_model():
    model = build_model(len(class_names))
    try:
        model.load_state_dict(torch.load('train6.pth', map_location='cpu'))
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()
DEVICE = torch.device("cpu")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
db_transform = torchaudio.transforms.AmplitudeToDB()

def calculate_real_snr(audio_path):
    """Calculate dynamic SNR from audio with time-series analysis"""
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(wav) == 0:
        raise ValueError("Empty audio file")
    total_duration = len(wav) / sr
    print(f"Audio loaded: {len(wav)} samples, {total_duration:.2f}s duration")
    
    noise_reference_duration = 0.3
    noise_samples = int(noise_reference_duration * sr)
    if len(wav) <= noise_samples:
        noise_samples = len(wav) // 3
    noise_reference = wav[:noise_samples]
    noise_power_ref = np.mean(noise_reference ** 2)
    
    signal_start = noise_samples
    signal_segment = wav[signal_start:]
    signal_power = np.mean(signal_segment ** 2) if len(signal_segment) > 0 else noise_power_ref
    print(f"Reference - Noise: {noise_power_ref:.6f}, Signal: {signal_power:.6f}")
    
    if noise_power_ref < 1e-10:
        noise_power_ref = 1e-10
    if signal_power <= noise_power_ref:
        ratio = max(1e-6, signal_power / noise_power_ref)
        overall_snr_db = 10 * math.log10(ratio)
    else:
        ratio = signal_power / noise_power_ref
        overall_snr_db = 10 * math.log10(ratio)
    overall_snr_db = max(-10.0, min(60.0, overall_snr_db))
    
    snr_time_series, time_bins = calculate_truly_dynamic_snr_time_series(wav, sr, total_duration)
    
    freqs, psd = signal.welch(wav, sr, nperseg=min(1024, len(wav)//4))
    print(f"OVERALL SNR: {overall_snr_db:.1f} dB")
    print(f"Time series: {len(snr_time_series)} points, range: {min(snr_time_series):.1f}-{max(snr_time_series):.1f} dB")
    
    return {
        'snr_db': float(overall_snr_db),
        'frequency_bins': freqs.tolist(),
        'power_spectrum': psd.tolist(),
        'snr_values_over_time': snr_time_series,
        'time_bins': time_bins,
        'total_duration': total_duration,
        'signal_power': float(signal_power),
        'noise_power': float(noise_power_ref)
    }

def calculate_truly_dynamic_snr_time_series(wav, sr, total_duration):
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    noise_reference_duration = 0.3
    noise_samples = int(noise_reference_duration * sr)
    if len(wav) <= noise_samples + frame_length:
        noise_samples = max(1, len(wav) // 4)
    noise_reference = wav[:noise_samples]
    noise_power_ref = np.mean(noise_reference ** 2)
    if noise_power_ref < 1e-10:
        noise_power_ref = 1e-10

    snr_values = []
    time_points = []
    for start in range(noise_samples, len(wav) - frame_length + 1, hop_length):
        end = start + frame_length
        frame = wav[start:end]
        signal_power = np.mean(frame ** 2)
        snr = 10 * np.log10(signal_power / noise_power_ref) if signal_power > 0 else -30
        snr_values.append(float(snr))
        time_points.append(float(start / sr))

    if time_points and time_points[0] > 0.0:
        snr_values = [snr_values[0]] + snr_values
        time_points = [0.0] + time_points

    return snr_values, time_points

def calculate_spectral_features(wav, sr):
    """Calculate spectral features from audio"""
    if len(wav) == 0:
        raise ValueError("Empty audio for spectral features")
    spectral_centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]
    centroid_mean = float(np.mean(spectral_centroid))
    spectral_rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(spectral_rolloff))
    zcr = librosa.feature.zero_crossing_rate(wav)[0]
    zcr_mean = float(np.mean(zcr))
    rms = librosa.feature.rms(y=wav)[0]
    rms_mean = float(np.mean(rms))
    print(f"Spectral Features - Centroid: {centroid_mean:.0f} Hz, Rolloff: {rolloff_mean:.0f} Hz, ZCR: {zcr_mean:.3f}, RMS: {rms_mean:.4f}")
    return {
        'spectral_centroid': centroid_mean,
        'spectral_rolloff': rolloff_mean,
        'zero_crossing_rate': zcr_mean,
        'rms_energy': rms_mean
    }

def get_snr_quality(snr_db):
    """Convert SNR to quality rating"""
    if snr_db >= 25:
        return "Excellent"
    elif snr_db >= 18:
        return "Good"
    elif snr_db >= 12:
        return "Fair"
    elif snr_db >= 6:
        return "Poor"
    else:
        return "Very Poor"

def fix_frames(mel, max_frames=FIXED_FRAMES):
    _, _, frames = mel.shape
    if frames < max_frames:
        mel = F.pad(mel, (0, max_frames - frames))
    elif frames > max_frames:
        mel = mel[:, :, :max_frames]
    return mel

def wav_to_mel(audio_path):
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    wav = torch.tensor(wav).unsqueeze(0)
    mel = mel_transform(wav)
    mel = db_transform(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    return fix_frames(mel, FIXED_FRAMES)

def sliding_windows(mel_tensor, win=FRAME_WINDOW_FRAMES, hop=FRAME_HOP_FRAMES):
    _, _, total = mel_tensor.shape
    starts = list(range(0, max(1, total - win + 1), hop))
    if not starts:
        starts = [0]
    for s in starts:
        e = min(s + win, total)
        yield s, e

def infer_windows(mel_tensor, model, device):
    results = []
    for s, e in sliding_windows(mel_tensor):
        chunk = mel_tensor[:, :, s:e]
        if chunk.shape[2] < FRAME_WINDOW_FRAMES:
            pad_amt = FRAME_WINDOW_FRAMES - chunk.shape[2]
            chunk = F.pad(chunk, (0, pad_amt))
        inp = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            pred_str = class_names[idx.item()]
            results.append((s, e, pred_str, conf.item(), probs.cpu()))
    return results

def frames_to_time_intervals(noisy_windows):
    if not noisy_windows:
        return []
    starts = [nw[0] for nw in noisy_windows]
    starts_sorted = sorted(starts)
    win_idx = [s // FRAME_HOP_FRAMES for s in starts_sorted]
    merged = []
    cur_start = win_idx[0]
    cur_end = cur_start
    for idx in win_idx[1:]:
        if idx - cur_end <= MERGE_GAP_FRAMES:
            cur_end = idx
        else:
            merged.append((cur_start, cur_end))
            cur_start = idx
            cur_end = idx
    merged.append((cur_start, cur_end))
    frame_duration = HOP_LENGTH / SAMPLE_RATE
    intervals = []
    for a, b in merged:
        start_frame = a * FRAME_HOP_FRAMES
        end_frame = b * FRAME_HOP_FRAMES + FRAME_WINDOW_FRAMES
        t0 = start_frame * frame_duration
        t1 = end_frame * frame_duration
        intervals.append((round(t0, 2), round(t1, 2)))
    return intervals

def render_spectrogram_with_noise(mel_np, noise_intervals, out_path):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    img = ax.imshow(mel_np, origin='lower', aspect='auto')
    frame_duration = HOP_LENGTH / SAMPLE_RATE
    for (t0, t1) in noise_intervals:
        f0 = t0 / frame_duration
        f1 = t1 / frame_duration
        rect = patches.Rectangle((f0, 0), f1 - f0, mel_np.shape[0],
                                 linewidth=0, edgecolor=None, facecolor='red', alpha=0.25)
        ax.add_patch(rect)
    ax.set_ylabel("Mel bins")
    ax.set_xlabel("Frames")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.title("Mel Spectrogram with Noise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def detect_sound_from_audio(audio_file):
    """Main function to detect sound from audio file"""
    try:
        if model is None:
            return {'success': False, 'error': 'Model not loaded'}
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            snr_analysis_data = calculate_real_snr(temp_path)
        except Exception as e:
            os.unlink(temp_path)
            return {'success': False, 'error': f'SNR calculation failed: {str(e)}'}

        spectral_features = {}
        try:
            wav, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
            spectral_features = calculate_spectral_features(wav, sr)
        except Exception as e:
            print(f"Spectral features skipped: {e}")
            spectral_features = {}

        mel = wav_to_mel(temp_path)
        os.unlink(temp_path)

        window_results = infer_windows(mel, model, DEVICE)

        noisy_windows = []
        all_probs = []
        for (s, e, pred, conf, probs) in window_results:
            all_probs.append(probs.numpy())
            if pred == "Unknown" or conf < FRAME_CONF_THRESH:
                noisy_windows.append((s, e, pred, conf))

        noise_intervals = frames_to_time_intervals(noisy_windows)
        noise_intervals = [[float(start), float(end)] for start, end in noise_intervals]

        mel_np = mel.squeeze(0).cpu().numpy()
        plot_name = f"annot_{uuid.uuid4().hex[:8]}.png"
        plot_path = os.path.join(PLOT_OUTPUT_DIR, plot_name)
        render_spectrogram_with_noise(mel_np, noise_intervals, plot_path)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx = int(np.argmax(avg_probs))
        clip_pred = class_names.get(top_idx, "Unknown")
        clip_conf = float(np.max(avg_probs))
        if clip_conf < CONFIDENCE_THRESHOLD:
            clip_pred = "Unknown"

        total_noise_duration = sum(end - start for start, end in noise_intervals)
        total_duration = snr_analysis_data.get('total_duration', 3.0)
        noise_percentage = (total_noise_duration / total_duration) * 100 if total_duration > 0 else 0
        signal_percentage = 100 - noise_percentage

        spectral_features_dict = {}
        if spectral_features:
            spectral_features_dict = {
                'spectral_centroid': float(spectral_features.get('spectral_centroid', 0)),
                'spectral_rolloff': float(spectral_features.get('spectral_rolloff', 0)),
                'zero_crossing_rate': float(spectral_features.get('zero_crossing_rate', 0)),
                'rms_energy': float(spectral_features.get('rms_energy', 0)),
            }

        snr_analysis_data.update({
            'quality': get_snr_quality(snr_analysis_data['snr_db']),
            'signal_percentage': round(signal_percentage, 1),
            'noise_percentage': round(noise_percentage, 1),
            'total_duration': total_duration,
            'signal_duration': round(total_duration - total_noise_duration, 2),
            'noise_duration': round(total_noise_duration, 2),
            'noise_segment_count': len(noise_intervals),
            'spectral_features': spectral_features_dict,
        })

        result = {
            'success': True,
            'predicted_class': clip_pred,
            'confidence': round(clip_conf * 100, 2),
            'noise_segments': noise_intervals,
            'spectrogram_plot': plot_path,
            'snr_analysis': snr_analysis_data,
            'all_predictions': {
                class_names[i]: round(float(avg_probs[i] * 100), 2)
                for i in range(len(class_names))
            }
        }

        print(f"FINAL SNR: {snr_analysis_data['snr_db']:.1f} dB, Quality: {snr_analysis_data['quality']}")
        print(f"Signal: {signal_percentage:.1f}%, Noise: {noise_percentage:.1f}%")

        return result

    except Exception as e:
        import traceback
        print("ERROR in sound detection:", traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'predicted_class': 'Unknown',
            'confidence': 0.0,
            'all_predictions': {},
        }