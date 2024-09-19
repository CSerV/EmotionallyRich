# EmotionallyRich


## Details

This project provides the code of using emotionally rich speech segments for depression prediction.

## Motivation

We think that emotionally rich speech segments provide more salient cues that can distinguish between depressed and non-depressed individuals.
In emotionally rich segments, such as those with high or low valence, differences in emotional expression between depressed and non-depressed individuals may be amplified, making depression easier to detect.

## Usage
To use this code, you first need to prepare your data, including extracting arousal, valence, and dominance scores, as well as PDEM embeddings, using the [public dimensional emotion model](https://github.com/audeering/w2v2-how-to). The scripts `1_extract_AVD_scores_from_PDEM.py` and `2_extract_embedding_from_PDEM.py` in the *Data_preparation* folder can be used as references. 

Then, you can follow the scripts in the *Train_and_Test* folder to train and test the model using your own data.


