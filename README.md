# Fake News Detection Using BERT and Human-Interpretable Features

## Overview
This repository is part of the **DS340W Research Project** at The Pennsylvania State University. It focuses on detecting fake news using a fine-tuned BERT model, with added human-interpretable features such as sentiment, post length, and readability. The research is centered on misinformation during the COVID-19 pandemic, utilizing a benchmark dataset of tweets to evaluate the effectiveness of hybrid models.

## Key Features
- **Dataset Preprocessing**: Includes text cleaning, label encoding, and tokenization.
- **Feature Engineering**:
  - Sentiment analysis with `TextBlob`.
  - Post length as word count.
  - Readability scoring via Flesch Reading Ease.
- **Model Architecture**:
  - Fine-tuned BERT model (`bert-base-uncased`) with additional custom layers.
  - Comparison between standalone BERT and hybrid models incorporating linguistic features.
- **Evaluation Metrics**: Includes accuracy, F1-score, precision, and recall.
- **Ablation Study**: Measures the impact of adding and removing engineered features.

## Installation
### Requirements
- Python 3.8 or higher
- Libraries:
  - `transformers`
  - `torch`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `textblob`
  - `textstat`

## Dataset
The dataset used in this study comprises **8,560 tweets** labeled as "real" or "fake." The dataset includes:
- Training set: 6,420 entries.
- Testing set: 2,140 entries.
Benchmarked by ML models, detailed in paper. 

## Results
- **Standalone BERT Model**:
  - F1-score: **0.84**
  - Balanced precision and recall for fake and real news.
- **Hybrid Model with Sentiment and Post Length**:
  - F1-score: **0.83**
  - Performance decline suggests feature redundancy, as BERT inherently captures sentiment and post length.

Confusion matrices and detailed metrics are available in the paper. 

## Running the Code 
All files are provided. Note, this was performed on Google Colab, so all files need to be added to google drive and minorly adjusted to be linked to your account. 

## Limitations
- **Language Dependency**: The study focuses solely on English tweets.
- **Platform-Specific**: Dataset is Twitter-specific and may not generalize to other platforms.
- **Excludes Multimedia**: No analysis of images or videos accompanying tweets.
