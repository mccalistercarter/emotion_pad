# PAD Emotion Project

## Build
docker build -t pad-emotion-recognition .

## Run
docker run --rm pad-emotion-recogntion

##  Overview
This project implements a real-time multimodal emotional recognition system for social human-robot interaction. The system will predict emotional state from a converstaional turn using the PAD (Pleasure, Arousal, Dominance) model. Verbal, vocal, and facial cues are processed and fused to produce continuous PAD values.

The full pipeline is designed to run end-to-end inside a docker container and to support real-time inference.

## PAD Model
The PAD model represents emotion along three continuous dimensions: pleasure, arousal, and dominance. In this project, each dimension is predicted as a continuous value within a configurable range.

## System Architecture
The system is organized as a modular pipeline that processes each modality independently before combining them for prediction. Verbal input is derived from text transcripts of user sepech, vocal input from audio waveforms (.wav) files, and facial input from live video (.mp4) frames or extracted face features. Each modality undergoes feature extraction before being passed through regressor models and finally through a fusion model that outputs final PAD predictions.

This modular design allows individual components to be modified or replaced wihtout changing the overall pipeline and also aids in the debugging process.

## Feature Extraction
Feature extraction is handled in a dedicated module for each modality found under the features/ directory. The system is designed to remain robust when one or more modalaties are missing or noisy.

### Text Features
Text features are extracted using a combination of pretrained transformer models to capture semantic, sentiment, and emotional infromation from the text inputs. A BERT-based encoder is used to generate sentence embeddings by averaging token-level states giving a contextual representation of the input text.

In addtion to semantic embeddings, sentiment polarity is computed with a sentiment classification model. The resulting score is mapped to a signed scalar to reflect postive or negative affect. Emotional content is further captured with an emotion classifier from which the highest-confidence emotion score is selected as a compact signal to describe the feature.

All features are concatenated into a single feature vector consisting of BERT embedding, sentiment score, and emotion confidence score. This vector is what is passed to the text-based PAD regressor.

### Audio Features
Audio-based features are extracted to capture prosodic and spectral characteristics of the aduio inputs while prioritizing real-time performance. Audio files are loaded and downsampled to a fixed sampling rate to reduce computation time. Stereo signals are converted to mono to simplify processing which was added in debugging to attempt to reduce time overhead.

Mel-frequency cepstral coefficients (MFCCs) are computed to represent the spectral features of the speech signal and their mean values across times are used to produce a fixed-size feature vector. Additional information is captured through energy and spectral centroid features which provide indicators to vocal intensity and frequency distribution. 

Pitch based features were initially considered but removed in an attempt to reduce impact on runtime. This was done to try and balance feature richness with latency results that were out of hand. All extracted features are concatenated into a single vector and passed to the audio-based regressor as was done with the text.

### Video Features
Video-based features are extracted to capture facial expressions and visual indicators of emotion throughout a conversational turn. Each video is processed by sampling frames at a reduced resolution to balance computational efficiency with emotional coverage. Frames are sampled at approximately one frame per second displaying another tradeoff between accuracy and real-time performance.

Facial emotion recognition is performed on sampled frames using a lighter-weight emotion detection model that outputs probablities across a fixed set of emotion categories. All successfully processed frames are averaging and aggregated into a single vector that summarizes the dominant facial emotions throughout the interaction.

Frames that fail emotion analysis are skipped and if no valid frames are detected, the system returns a zero-valued feature vector. this helps to keep the run-time and simplicity down. With additional time and optimization, strategies could be improved to enhance accuracy, model complexity, and reduce latency.

## Modeling and Regresson
Seperate regression models are implemented for each modality, text, audio, and video, to map modality-specific features to continuous PAD values. This design allows for the system to stay modular and interpretable for aiding in the testing and debugging process.

Each regressor produces and intermediate PAD prediction based solely on its resepctive modality input. These predictions servce as inputs to the final fusion stage rather than directly producing the system output.

Simple linear regression is used to map the extracted features to continuous PAD values. The LinearRegression class from the scikit-learn library is used to fit each models by cmoparing input features to PAD labels, producing a mapping that can generate predictions across inference. These trained regressors are invoked in both the infer.py and demo_script.py scripts to provide modality-specific PAD estimtes, that are combined in the late fusion stage. While linear regression provides a good baseline, further improvement could be made with time and experimentation to support more sophisticated models or neural architectures.

## Multimodal Fusion
The final PAD prediction is produced using a late fusion strategy, where the outputs of the individual regressors are combined into a single estimate. The current implementation uses a simple and computationally efficient fusion model, prioritizing robustness and real-time performance.

In this approach each regressor contributes equally and any missing or empty modality features are replaced with zeros to prevent errors and ensure robustness. With additional time and experimentation, this fusion stage could be extended to incorporate more sophisticated regression techniques, learned weighting schemes, or temporal modeling across multiple converstional turns.

## Performance and Latency
** Insert information about latency evaluation and prediction accuracy stats

Meeting the real-time latency constraint proved challenging, with feature extraction accounting for the majority, or almost all of the runtime. While inference itself is lightweight, the computational cost of extracting features from text, audio, and video significantly impacts overall responsiveness. With additional research and development time, latency could be improved through optimized feature pipelines, caching, or alternative more lightweight feature representations.

## Limitations and Future Improvements
The current system represents a functional and extensible baseline for multimodal PAD prediction. However, emotion recognition remains sensitive to input noise, subjectivity in intial PAD labelling, and the quality of extracted features. Additional time and testing would allow for more extensive model tuning, improved fusion strategies, stronger regressors, and systematic latency optimization.

Future work could also explore context across multiple turns, personalized models, and more advanced multimodal representations to improve both accuracy and responsiveness. I would also like to see how the current model performs with more training and some simple fine-tuning.