# Flatiron Module 5 Project - Classification
*Bailey Bjornstad*

In this project, we are tasked with a comparison of various classification systems on datasets of our choosing. I have chosen to work with a dataset of different bird vocalizations, found on Kaggle.

## The Data
The data contains 2730 different samples of bird vocalizations in MP3 format from birds in California ands Nevada. The task is to classify the birds into their species, one of ??. However, the MP3 files definitely need some processing in order to create satisfactory features on which classification depends. In particular, we will need to perform feature extraction on these files, using various methods designed for audio data.

## Project Structure
- Raw MP3 files are not saved on git, but can be found on Kaggle.
- The processing module is taken from my other repository called [audio-feature-extraction](https://github.com/bbjornstad/audio-feature-extraction), and are contained in the `afe` folder. Documentation about their usage can be found at that repository or through the accompanying blog post.
- The processed dataframes are stored in [`feature_extraction`](./feature_extraction/).
- Notebook Files to handle the actual usage of the processing classes for feature extraction and analysis.

## Dependencies
- The feature extraction process relies on `librosa`. Please make sure you install it with the following
  command:
  - `conda install -c conda-forge librosa`
  - or `pip install librosa`

## Next Steps
- Primarily, we want to use some of the newly updated features from the audio-feature-extraction project to handle some processing of the audio to make classification better.
    - We have the ability to now preemphasis filter, bandpass filter, and onset detect. This should help us to normalize some of the admittedly variable data.
    - We may need to think about implementing some form of detection of pattern repetition to further this process.
