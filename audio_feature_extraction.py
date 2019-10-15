import pandas as pd
import numpy as np
import librosa


class AudioFeatureExtractor:
    def __init__(
            self,
            sr=22050,
            window_size=512,
            n_mfcc=40):
        self.sr = sr
        self.n_fft = window_size
        self.hop_length = int(self.n_fft / 4)
        self.n_mfcc = n_mfcc

    def get_audio(self, file_path):
        x, sr = librosa.load(file_path, sr=self.sr)
        return x

        # -----
        # Feature Extraction
        # -----
    def extract_chroma_stft(self, audio):
        """
        Extracts the chromagram from the given audio
        This is a lot like a spectrogram except we are mapping
        onto the chromatic scale.

        """
        chroma_stft = librosa.feature.chroma_stft(
            audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma_stft

    def extract_chroma_cqt(self, audio):
        chroma_cqt = librosa.feature.chroma_cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cqt

    def extract_chroma_cens(self, audio):
        chroma_cens = librosa.feature.chroma_cens(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cens

    def extract_melspectrogram(self, audio):
        melspectrogram = librosa.feature.chroma_cens(
            audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length)
        return melspectrogram

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length)
        return mfcc

    def extract_rms(self, audio):
        pass

    def extract_spectral_centroid(self, audio):
        pass

    def extract_spectral_bandwidth(self, audio):
        pass

    def extract_spectral_contrast(self, audio):
        pass

    def extract_spectral_flatness(self, audio):
        pass

    def extract_spectral_rolloff(self, audio):
        pass

    def extract_poly_features(self, audio):
        pass

    def extract_tonnetz(self, audio):
        pass

    def extract_zero_crossing_rate(self, audio):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length)
        return zero_crossing_rate

    def extract_tempogram(self, audio):
        pass

    def extract_fourier_tempogram(self, audio):
        pass

    def batch_extract_feature(
            self,
            file_index,
            extraction_method,
            audio_folder='raw_data/',
            audio_index='bird_vocalization_index.csv',
            average=True):
        index_df = pd.read_csv(audio_index, index_col=0)
        for file_name in index_df.file_name:
            audio = self.get_audio(audio_folder + file_name)
            feature_mat = extraction_method(audio)
            if average:
                feature_mat = np.mean(feature_mat, axis=1)
                feature_series = pd.Series(feature_mat.T, name=file_name[:-4])
                return feature_series
        return feature_mat

        # -----
        # Feature Manipulation
        # -----
    def feature_delta(self, audio):
        pass

    def feature_stack_delta(self, audio):
        pass

    # -----
    # Feature Inversion
    # -----
    def inverse_mel_to_stft(self, melspectrogram):
        pass

    def inverse_mel_to_audio(self, melspectrogram):
        pass

    def inverse_mfcc_to_mel(self, mfcc):
        pass

    def inverse_mfcc_to_audio(self, mfcc):
        pass
