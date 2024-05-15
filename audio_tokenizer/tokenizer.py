from typing import*
import numpy as np
from scipy.io import wavfile

class AudioTokenizer:
    def __init__(self, sample_rate: int = 44100, max_freq: int = 16384,
                 tensor_type: np.dtype = np.float16, max_input_len: int = 10, step: int = 1) -> None:
        self.sample_rate = sample_rate
        self.max_freq = max_freq
        self.tensor_type = tensor_type
        self.step = step  # Duration of each chunk in seconds
        self.max_input_len = max_input_len  # Duration in seconds of minimum total audio processed

    def tokenize(self, file_path: str) -> Tuple[np.ndarray, int]:
        rate, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        step_samples = self.sample_rate * self.step
        total_length = ((len(data) // step_samples) + 1) * step_samples
        data = np.pad(data, (0, total_length - len(data)), 'constant')
        chunks = []
        end = self.max_input_len*self.sample_rate if len(data) > self.max_input_len*self.sample_rate else len(data)
        for start in range(0, end, step_samples):
            chunk_data = data[start:start + step_samples]
            fft_data = np.fft.fft(chunk_data)[:self.max_freq]
            max_magnitude = 32767  # Using a fixed number to normalize the magnitude
            if max_magnitude == 0:
                max_magnitude = 0.001  # Prevent division by zero
            magnitudes = np.abs(fft_data) / max_magnitude
            phases = (np.angle(fft_data) + np.pi) / (2 * np.pi)
            flat_array = np.concatenate([magnitudes, phases]).astype(self.tensor_type)
            chunks.append(flat_array)
        # Ensure the total audio length meets the minimum required length
        required_chunks = self.max_input_len // self.step
        while len(chunks) < required_chunks:
            zero_chunk = np.zeros((2 * self.max_freq,), dtype=self.tensor_type)  # Pad with zeros
            chunks.append(zero_chunk)
        return np.array(chunks), rate

    def detokenize(self, frequency_data: np.ndarray, rate: int, output_file: str):
        reconstructed_audio = np.zeros(0)
        for data in frequency_data:
            magnitudes = data[:self.max_freq]
            phases = data[self.max_freq:] * 2 * np.pi - np.pi
            complex_spectrum = magnitudes * np.exp(1j * phases)
            N = self.sample_rate * self.step
            full_spectrum = np.zeros(N, dtype=complex)
            full_spectrum[:len(complex_spectrum)] = complex_spectrum
            full_spectrum[-len(complex_spectrum)+1:] = np.conj(complex_spectrum[1:][::-1])
            audio_chunk = np.fft.ifft(full_spectrum).real
            reconstructed_audio = np.concatenate((reconstructed_audio, audio_chunk))
        max_abs = np.max(np.abs(reconstructed_audio)) or 0.001  # Use or to prevent division by zero
        reconstructed_audio = np.int16(reconstructed_audio / max_abs * 32767)
        wavfile.write(output_file, rate, reconstructed_audio)

if __name__ =="__main__":
    tokenizer = AudioTokenizer(step=1, tensor_type=np.float16, max_freq=512*16*2, max_input_len=60*2)
    frequency_data, sample_rate = tokenizer.tokenize('audio_data/input_2.wav')
    #print(frequency_data)
    tokenizer.detokenize(frequency_data, sample_rate, 'reconstructed_audio.wav')
