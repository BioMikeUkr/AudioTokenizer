from typing import*
import numpy as np
from scipy.io import wavfile

class SimpleAudioTokenizer:
    def __init__(self, sample_rate: int = 44100, max_freq: int = 16384,
                 tensor_type: np.dtype = np.float16, max_input_len: int = 10, step: int = 1) -> None:
        self.sample_rate = sample_rate
        self.max_freq = max_freq
        self.tensor_type = tensor_type
        self.step = step  # Duration of each chunk in seconds
        self.max_input_len = max_input_len  # Duration in seconds of the minimum total audio processed

    def tokenize(self, file_path: str) -> Tuple[dict, int]:
        rate, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        step_samples = self.sample_rate * self.step
        max_samples = self.max_input_len * self.sample_rate  # Max samples to consider for the entire file processing
        data = data[:max_samples]  # Limit data to max_input_len seconds for magnitude calculation

        # Calculate the total length required for padding
        total_length = ((len(data) // step_samples) + 1) * step_samples
        data = np.pad(data, (0, total_length - len(data)), 'constant')

        # Perform FFT on all data initially to determine the global maximum magnitude
        full_fft_data = np.fft.fft(data)
        max_magnitude = np.max(np.abs(full_fft_data[:self.max_freq])) if np.max(
            np.abs(full_fft_data[:self.max_freq])) != 0 else 1

        # Generate magnitudes and phases for each chunk
        magnitudes, phases = [], []
        for start in range(0, len(data), step_samples):
            chunk_data = data[start:start + step_samples]
            fft_data = np.fft.fft(chunk_data)[:self.max_freq]
            magnitudes.append((np.abs(fft_data) / max_magnitude).astype(self.tensor_type))
            phases.append(((np.angle(fft_data) + np.pi) / (2 * np.pi)).astype(self.tensor_type))

        # Padding with zeros to ensure the minimum required length
        required_chunks = (self.max_input_len // self.step) - len(magnitudes)
        for _ in range(required_chunks):
            zero_chunk = np.zeros(self.max_freq, dtype=self.tensor_type)
            magnitudes.append(zero_chunk)
            phases.append(zero_chunk)

        return {'magnitudes': np.array(magnitudes), 'phases': np.array(phases)}, rate
    
    def detokenize(self, tokenized_audio: dict, rate: int, output_file: str):
        reconstructed_audio = np.zeros(0)
        for magnitudes, phases in zip(tokenized_audio['magnitudes'], tokenized_audio['phases']):
            phases = phases * 2 * np.pi - np.pi  # Convert back from [0, 1] to [-pi, pi] range
            complex_spectrum = magnitudes * np.exp(1j * phases)
            N = self.sample_rate * self.step
            full_spectrum = np.zeros(N, dtype=complex)
            full_spectrum[:len(complex_spectrum)] = complex_spectrum
            full_spectrum[-len(complex_spectrum)+1:] = np.conj(complex_spectrum[1:][::-1])
            audio_chunk = np.fft.ifft(full_spectrum).real
            reconstructed_audio = np.concatenate((reconstructed_audio, audio_chunk))
        max_abs = np.max(np.abs(reconstructed_audio)) or 1  # Use or to prevent division by zero
        reconstructed_audio = np.int16(reconstructed_audio / max_abs * 32767)
        wavfile.write(output_file, rate, reconstructed_audio)

if __name__ =="__main__":
    tokenizer = SimpleAudioTokenizer(step=1, tensor_type=np.float16, max_freq=512*16*2, max_input_len=60*2)
    frequency_data, sample_rate = tokenizer.tokenize('audio_data/input_2.wav')
    print(frequency_data)
    tokenizer.detokenize(frequency_data, sample_rate, 'reconstructed_audio.wav')
