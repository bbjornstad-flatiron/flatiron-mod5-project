import pandas as pd
import numpy as np
import librosa


class AudioFeatureExtractor:
    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            n_mfcc=20):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = int(self.frame_length / 4)
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
            n_fft=self.frame_length,
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
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        return melspectrogram

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        return mfcc

    def extract_rms(self, audio):
        rms = librosa.feature.rms(
            audio,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return rms

    def extract_spectral_centroid(self, audio):
        spectral_centroid = librosa.feature.spectral_centroid(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_centroid

    def extract_spectral_bandwidth(self, audio):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        return spectral_bandwidth

    def extract_spectral_contrast(self, audio):
        spectral_contrast = librosa.feature.spectral_contrast(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_contrast

    def extract_spectral_flatness(self, audio):
        spectral_flatness = librosa.feature.spectral_flatness(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_flatness

    def extract_spectral_rolloff(self, audio):
        spectral_rolloff = librosa.feature.spectral_rolloff(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )

    def extract_poly_features(self, audio, poly_order=3):
        poly_features = librosa.feature.poly_features(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            order=poly_order
        )
        return poly_features

    def extract_tonnetz(self, audio):
        tonnetz = librosa.feature.tonnetz(
            audio,
            sr=self.sr
        )
        return tonnetz

    def extract_zero_crossing_rate(self, audio):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length)
        return zero_crossing_rate

    def extract_tempogram(self, audio):
        pass

    def extract_fourier_tempogram(self, audio):
        pass

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


class BatchExtractor:
    def __init__(
        self,
        sr=22050,
        frame_length=1024,
        n_mfcc=20,
        audio_folder='raw_data/',
        audio_index='bird_vocalization_index.csv'):
        
        self.audio_folder = audio_folder
        self.audio_index = pd.read_csv(audio_index, index_col=0)
        self.sr = sr
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        self.afe = AudioFeatureExtractor(self.sr, self.frame_length, self.n_mfcc)
        # contains associations between string short forms and extraction methods in the
        # objects AudioFeatureExtractor
        self.extraction_dict = {'mfcc': self.afe.extract_mfcc, 
                                'melspec': self.afe.extract_melspectrogram,
                                'zcr': self.afe.extract_zero_crossing_rate,
                                'ccqt': self.afe.extract_chroma_cqt,
                                'cstft': self.afe.extract_chroma_stft,
                                'ccens': self.afe.extract_chroma_cens,
                                'rms': self.afe.extract_rms,
                                'centroid': self.afe.extract_spectral_centroid,
                                'bandwidth': self.afe.extract_spectral_bandwidth,
                                'contrast': self.afe.extract_spectral_contrast,
                                'flatness': self.afe.extract_spectral_flatness,
                                'rolloff': self.afe.extract_spectral_rolloff,
                                'poly': self.afe.extract_poly_features,
                                'tonnetz': self.afe.extract_tonnetz}
        
    def batch_extract_feature(self, extraction_method, results_folder='feature_extraction/'):
        method = self.extraction_dict[extraction_method]
        
        for file_name in self.audio_index.file_name:
            audio = self.afe.get_audio(self.audio_folder+file_name)
            feature_matrix = pd.DataFrame(method(audio).T)
            n_cols = len(feature_matrix.columns)
            feature_cols = [f'{extraction_method}_{i}' for i in range(n_cols)]
            feature_matrix.columns = feature_cols
            
            name = file_name[:-4]
            
            feature_matrix.to_csv(f'{results_folder}{name}_{extraction_method}_features.csv', index=False)
            
    def merge_features(self, features_to_merge, results_folder='feature_extraction/'):
        for file_name in self.audio_index.file_name:
            name = file_name[:-4]
            sample_df = pd.DataFrame()
            for feature in features_to_merge:
                feature_df = pd.read_csv(f'{results_folder}{name}_{feature}_features.csv')
                sample_df = pd.concat([sample_df, feature_df], axis=1)
            sample_df.to_csv(f'{results_folder}{name}_merged_features.csv', index=False)
            
    def batch_extract_and_merge(self, features_to_merge, results_folder='feature_extraction/'):
        pass