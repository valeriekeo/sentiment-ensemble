# src/pipeline.py
# Pipeline for sentiment ensemble + NER triage
# Wraps VADER, RoBERTa, and spaCy into clean callable functions

import torch
import torch.nn.functional as F
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import json
import re

# Use RoBERTa for Sentiment Analysis
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Use HuggingFace NER Model
NER_MODEL = "dslim/bert-base-NER"
LABELS = {0: 'negative', 1: 'neutral', 2: 'positive'}

# F1-macro weights from 1000 tweet test set evaluation
# I drop Llama3 as it is not supported by HuggingFace Spaces
MODEL_WEIGHTS = 