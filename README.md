# Flatiron Module 5 Project - Classification
*Bailey Bjornstad*

In this project, we are tasked with a comparison of various classification systems on datasets of our choosing. I have chosen to work with a dataset of different bird vocalizations, found on Kaggle.

## The Data
The data contains 2730 different samples of bird vocalizations in MP3 format from birds in California ands Nevada. The task is to classify the birds into their species, one of ??. However, the MP3 files definitely need some processing in order to create satisfactory features on which classification depends. In particular, we will need to perform feature extraction on these files, using various methods designed for audio data.

## Project Structure
- Raw MP3 files are not saved on git, but can be found on Kaggle.
- The processing script is contained in [audio_feature_extraction.py](audio_feature_extraction.py).
- The processed dataframes are stored in [`feature_extraction`](./feature_extraction/).
- Notebook Files:

## Dependencies
- The feature extraction process relies on `librosa`. Please make sure you install it with the following
  command:
  - `conda install -c conda-forge librosa`
  - or `pip install librosa`
