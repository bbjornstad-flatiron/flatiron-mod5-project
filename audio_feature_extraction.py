import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioFeatureExtractor:
    """
    This class implements an extraction object for audio samples.
    Mostly this is a wrapper for the librosa module, which has low level
    implementations of feature extraction. In particular it standardizes
    a set of parameters for the feature extraction process so that all
    samples analyzed with a particular instance have consistent framing and
    features.

    Attributes:
    -----------
        :int sr:            desired sample rate of audio
        :int frame_length:  an integer power of 2 representing the length of
                            fft and other windows
        :int hop_length:    computed from the frame length, represents the 
                            length that windows jump (currently 1/4 a window)
        :int n_mfcc:        the desired number of Mel-frequency cepstral
                            coefficients to compute
        :int fmin:          the minimum frequency used to compute certain 
                            features
        :int fmax:          the maximum frequency used to compute certain
                            features
    """

    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            n_mfcc=20,
            fmin=1024,
            fmax=8192):
        """
        Initializes an AudioFeatureExtractor object

        Parameters:
        -----------
            :int sr:            integer for the desired sample rate
                                (default 22050)
            :int frame_length:  an integer power of two for the frame length to
                                be used in feature extraction (default 1024)
            :int n_mfcc:        the number of Mel-frequency cepstral
                                coefficients to compute (default 20)
            :int fmin:          integer for the lowest frequency that will be
                                computed in certain features (default 1024)
            :int fmax:          integer for the highest frequency that will be
                                computed in certain features.
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = int(self.frame_length / 4)
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax

    def get_audio(self, file_path):
        """
        Gets audio as a numpy array with the object's sample rate from the 
        given string file path.
        """
        x, sr = librosa.load(file_path, sr=self.sr)
        return x

        # -----
        # Feature Extraction
        # -----
    def extract_stft(self, audio):
        """
        Extracts the short term fourier transform of the given audio, a process
        in which an audio sample is chunked into frames and the associated
        frequency energy content is analyzed, thus transforming from the time
        domain to the frequency domain (in timed chunks).
        """
        stft = librosa.stft(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        return stft

    def extract_chroma_stft(self, audio):
        """
        Extracts a chromagram of the given audio, like a spectrogram but binned 
        into the chromatic scale.
        """
        chroma_stft = librosa.feature.chroma_stft(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return chroma_stft

    def extract_chroma_cqt(self, audio):
        """
        Extracts a constant-Q chromagram of the given audio.
        """
        chroma_cqt = librosa.feature.chroma_cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cqt

    def extract_chroma_cens(self, audio):
        """
        Extracts an Energy Normalized chromagram of the given audio.
        """
        chroma_cens = librosa.feature.chroma_cens(
            audio,
            sr=self.sr,
            hop_length=self.hop_length)
        return chroma_cens

    def extract_melspectrogram(self, audio):
        """
        Extracts a Mel-windowed spectrogram of the given audio.
        """
        melspectrogram = librosa.feature.melspectrogram(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax)
        return melspectrogram

    def extract_mfcc(self, audio):
        """
        Extracts a number of Mel-frequency cepstral coefficients from the
        given audio, where the number is controlled as an object attribute.
        """
        mfcc = librosa.feature.mfcc(
            audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax)
        return mfcc

    def extract_rms(self, audio):
        """
        Extracts the root-mean-square value for each frame of the given audio.
        """
        rms = librosa.feature.rms(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return rms

    def extract_spectral_centroid(self, audio):
        """
        Extracts the spectral centroid of the given audio.
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_centroid

    def extract_spectral_bandwidth(self, audio):
        """
        Extracts the spectral bandwidth of the given audio.
        """
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        return spectral_bandwidth

    def extract_spectral_contrast(self, audio):
        """
        Extracts the spectral contrast of the given audio.
        """
        spectral_contrast = librosa.feature.spectral_contrast(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_contrast

    def extract_spectral_flatness(self, audio):
        """
        Extracts the spectral flatness of the given audio.
        """
        spectral_flatness = librosa.feature.spectral_flatness(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_flatness

    def extract_spectral_rolloff(self, audio):
        """
        Extracts the spectral rolloff of the given audio.
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return spectral_rolloff

    def extract_poly_features(self, audio, poly_order=3):
        """
        Extracts polynomial features from the spectrogram of the given audio,
        using the optionally specified poly_order parameter to control the
        degree (default 3).
        """
        poly_features = librosa.feature.poly_features(
            audio,
            sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            order=poly_order
        )
        return poly_features

    def extract_tonnetz(self, audio):
        """
        Extracts the tonnetz (tonal centroid features) from the given audio.
        """
        tonnetz = librosa.feature.tonnetz(
            audio,
            sr=self.sr
        )
        return tonnetz

    def extract_zero_crossing_rate(self, audio):
        """
        Extracts the zero crossing rate from the given audio.
        """
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length)
        return zero_crossing_rate

    def extract_tempogram(self, audio):
        """
        Extracts the tempogram from the given audio.

        ---Currently Unimplemented---
        """
        pass

    def extract_fourier_tempogram(self, audio):
        """
        Extracts the fourier tempogram from the given audio.

        ---Currently Unimplemented---
        """
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
    """
    This class implements an object that can handle batch extraction of a set
    of audio samples. In particular, at instantiation a desired folder of audio
    and associated metadata index can be specified. By default, a BatchExtractor
    will look for audio files in the `raw_data` folder, and the index file is
    specified as a CSV file and defaults to `bird_vocalization_index.csv` Note
    that the index can also be specified as an already instantiated dataframe.

    Attributes:
    -----------
        :int sr:                    desired sample rate for extraction processes
                                    as an integer
        :int frame_length:          integer 2^n representing desired windowing 
                                    length for feature extraction processes
        :int n_mfcc:                integer for the desired number of 
                                    Mel-windowed cepstral coefficients to be
                                    extracted
        :AudioFeatureExtractor afe: an AudioFeatureExtractor object that handles
                                    the actual extraction for each sample
        :dict extraction_dict:      a dictionary association between string
                                    abbreviations of features and the associated
                                    extraction methods in the afe
        :str audio_folder:          a string representing a path to a folder
                                    containing raw audio files
        :df or str audio_index:     a dataframe or path to CSV that can be read
                                    as a dataframe, representing metadata
                                    and labeling information for each audio
                                    sample in the audio_folder
        :int fmin:                  integer representing the minimum frequency
                                    that will be computed for certain spectral
                                    based features.
        :int fmax:                  integer representing the maximum frequency
                                    that will be computed for certain spectral
                                    based features.
    """

    def __init__(
            self,
            sr=22050,
            frame_length=1024,
            n_mfcc=20,
            audio_folder='raw_data/',
            audio_index='bird_vocalization_index.csv',
            fmin=1024,
            fmax=8192):
        """
        Initializes a BatchExtractor object

        Parameters:
        -----------
            :int sr:                An integer sample rate for analysis of all 
                                    files (default 22050)
            :int frame_length:      An integer power of two representing the
                                    frame length of windows used in extraction
                                    as a number of samples (default 1024)
            :int n_mfcc:            An integer for the number of Mel-frequency
                                    cepstral coefficients to compute (default
                                    20)
            :str audio_folder:      A string representing a path to a folder
                                    containing audio samples.
            :df or str audio_index: a dataframe or path to CSV that can be read
                                    as a dataframe and holding a metadata index
                                    for the files in the audio_folder. Must have
                                    a file_name column identifying the file
                                    location of the MP3 relative to the
                                    audio_folder.
            :int fmin:              integer representing the minimum frequency
                                    to be used for computation of certain
                                    spectral based features (default 1024)
            :int fmax:              integer representing the maximum frequency
                                    to be used for computation fo certain
                                    spectral based features (default 8192)
        """

        self.audio_folder = audio_folder
        self.audio_index = None

        # check type on the index parameter to see if it needs to be read in.
        if isinstance(audio_index, str):
            self.audio_index = pd.read_csv(audio_index, index_col=0)
        else:
            self.audio_index = audio_index

        self.sr = sr
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax

        # intialize an underlying AudioFeatureExtractor with the given params.
        self.afe = AudioFeatureExtractor(
            self.sr, self.frame_length, self.n_mfcc, self.fmin, self.fmax)

        # contains associations between string short forms and extraction
        # methods in the object's AudioFeatureExtractor
        self.extraction_dict = {'stft': self.afe.extract_stft,
                                'mfcc': self.afe.extract_mfcc,
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
                                'tonnetz': self.afe.extract_tonnetz,
                                'poly': self.afe.extract_poly_features
                                }

    def batch_extract_feature(
            self,
            extraction_method,
            results_folder='feature_extraction/'):
        """
        Extracts a single feature from all of the audio files in the
        audio_folder.

        Does not return a value, but rather saves the extracted features as
        CSV files.

        Parameters:
        -----------
            :str extraction_method: a string represnting a key in the
                                    extraction_dict attribute, in other words
                                    the defined abbreviation for the desired
                                    feature
            :str results_folder:    a string representing a file path to save
                                    the extracted features (default 
                                    `feature_extraction/`)
        """
        method = self.extraction_dict[extraction_method]

        for file_name in self.audio_index.file_name:
            audio = self.afe.get_audio(self.audio_folder + file_name)
            feature_matrix = pd.DataFrame(method(audio).T)
            n_cols = len(feature_matrix.columns)
            feature_cols = [f'{extraction_method}_{i}' for i in range(n_cols)]
            feature_matrix.columns = feature_cols

            name = file_name[:-4]

            feature_matrix.to_csv(
                f'{results_folder}{name}_{extraction_method}_features.csv',
                index=False)

    def batch_extract_features(
            self,
            extraction_methods,
            results_folder='feature_extraction/'):
        """
        Extracts a list of features from all the audio samples in the
        audio_folder.

        Does not return a value, but rather saves the extracted features as
        CSV files.

        Parameters:
        -----------
            :list(str) extraction_methods:  a list of strings representing
                                            abbreviations for the desired
                                            features to extract
            :str results_folder:            a string representing a file path to 
                                            save the extracted features (default 
                                            `feature_extraction/`)
        """
        for method in extraction_methods:
            print(method)
            self.batch_extract_feature(method, results_folder=results_folder)

    def merge_features(
            self,
            features_to_merge,
            results_folder='feature_extraction/'):
        """
        Merges a list of features found as extracted CSV files for each sample
        in the audio_folder.

        Parameters:
        -----------
            :list(str) features_to_merge:   a list of strings representing
                                            abbreviations for the desired
                                            features to merge
            :str results_folder:            a string representing a file path to 
                                            find the extracted features and save
                                            the merged features (default 
                                            `feature_extraction/`)
        """
        for file_name in self.audio_index.file_name:
            name = file_name[:-4]
            sample_df = pd.DataFrame()
            for feature in features_to_merge:
                feature_df = pd.read_csv(
                    f'{results_folder}{name}_{feature}_features.csv')
                sample_df = pd.concat([sample_df, feature_df], axis=1)
            sample_df.to_csv(
                f'{results_folder}{name}_merged_features.csv', index=False)

    def batch_extract_and_merge(
            self,
            extraction_methods,
            results_folder='feature_extraction/'):
        """
        Combines the batch feature extraction method and the merging method.
        """
        self.batch_extract_features(
            extraction_methods,
            results_folder=results_folder)
        self.merge_features(extraction_methods, results_folder=results_folder)

    def merge_and_flatten_features(
            self,
            extraction_methods,
            results_folder='feature_extraction/',
            label=False):
        """
        Merges the given list of features into a flattened dataframe, where each
        row represents all of the feature data for each frame for a given
        sample, and where the rows with fewer frames have those corresponding
        columns set to 0.

        Parameters:
        -----------
            :list(str) extraction_methods:  a list containing string 
                                            abbreviations for the features which
                                            we want to merge and flatten into
                                            a single dataframe.
            :str results_folder:            a string indicating a path to a
                                            folder containing feature matrices
                                            to merge
            :bool label:                    a boolean indicating whether or not
                                            each row should be labeled with its
                                            associated target.

        Returns:
        --------
            :(n, frames*n_feats) df:        a dataframe indexed by the name of
                                            the sample (derived from file name)
                                            and containing columns of the form
                                            feat_{attr}_{frame} where attribute
                                            represents a single attribute of a
                                            given feature and frame represents
                                            the frame number.
        """
        flattened_df = pd.DataFrame()

        for method in extraction_methods:
            method_df = pd.DataFrame()
            max_frames = 0
            for file_name in self.audio_index.file_name:
                name = file_name[:-4]
                feature_matrix = pd.read_csv(
                    f'{results_folder}{name}_{method}_features.csv')

                if len(feature_matrix.index) > max_frames:
                    max_frames = len(feature_matrix.index)

                col_names = list(feature_matrix.columns)
                new_row = np.ravel(feature_matrix.to_numpy(), order='F')
                new_series = pd.Series(new_row)
                new_series.name = name
                method_df = method_df.append(new_series)

            method_df.columns = [
                f'{col_name}_{t}' for col_name in col_names for t in range(max_frames)
            ]

            flattened_df = pd.concat([flattened_df, method_df], axis=1)

        if label:
            flattened_df['label'] = self.audio_index.label
        return flattened_df.fillna(0)


class FeatureVisualizer:
    def __init__(
            self,
            feature_folder='feature_extraction/',
            default_figure_size=(18, 8)):
        self.default_figure_size = default_figure_size
        self.feature_folder = feature_folder

    def plot_melspec(self, sample_name):
        melspec_fig = plt.figure(figsize=self.default_figure_size)
        try:
            melspec = pd.read_csv(
                f'{self.feature_folder}{sample_name}_melspec_features.csv').T
            melspec = melspec.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        librosa.display.specshow(melspec_db, x_axis='time', y_axis='mel')
        plt.title(f'Melspectrogram -- {sample_name}')
        return melspec_fig

    def plot_chromagram(self, sample_name):
        cstft_fig = plt.figure(figsize=self.default_figure_size)
        try:
            cstft = pd.read_csv(
                f'{self.feature_folder}{sample_name}_cstft_features.csv').T
            cstft = cstft.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        librosa.display.specshow(cstft, x_axis='time', y_axis='chroma')
        plt.title(f'Chromagram -- {sample_name}')
        return cstft_fig

    def plot_spectrogram(self, sample_name):
        stft_fig = plt.figure(figsize=self.default_figure_size)
        try:
            stft = pd.read_csv(
                f'{self.feature_folder}{sample_name}_stft_features.csv').T
            stft = stft.to_numpy()
        except FileNotFoundError:
            print('Feature matrix not found...did you remember to extract the features?')
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(stft_db, y_axis='linear')
        plt.title(f'Spectrogram -- {sample_name}')
        return stft_fig
