# src/pipeline.py
# Pipeline for sentiment ensemble + NER triage
# Wraps VADER, RoBERTa, and HuggingFace NER into clean callable functions

import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import InferenceClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import re
import os

# Use RoBERTa for Sentiment Analysis
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Use HuggingFace NER Model
NER_MODEL = "dslim/bert-base-NER"
LABELS = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Llama 3 via HuggingFace Inference API is used as tiebreaker on LOW confidence predictions
# I drop spaCy for preliminary demo and will consider returning to add it later

# F1-macro weights from 1000 tweet test set evaluation, kept for reference
MODEL_WEIGHTS = {
    'vader': 0.529,
    'roberta': 0.724
}

# Triage thresholds
ROBERTA_CONFIDENCE_THRESHOLD = 0.70   # below this = low confidence
VADER_DISAGREEMENT_THRESHOLD = 0.20   # above this = strong VADER disagreement

def load_models(device=None, hf_token = None):
    """Load all models into memory. Call once at startup."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    models = {}

    models['hf_token'] = hf_token  # store for LLM calls
    
    print("Loading VADER...")
    models['vader'] = SentimentIntensityAnalyzer()
    
    print("Loading RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    roberta = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    roberta = roberta.to(device)
    roberta.eval()
    models['roberta'] = roberta
    models['tokenizer'] = tokenizer
    models['device'] = device
    
    print("Loading HuggingFace NER...")
    models['ner'] = pipeline(
        'ner',
        model=NER_MODEL,
        aggregation_strategy='first',
        device=0 if device == 'cuda' else -1
    )
    
    print("All models loaded.")
    return models

def run_vader(text, sia):
    """Run VADER sentiment classification on a single tweet."""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        label = 'positive'
        probs = [0.1, 0.1, 0.8]
    elif compound <= -0.05:
        label = 'negative'
        probs = [0.8, 0.1, 0.1]
    else:
        label = 'neutral'
        probs = [0.1, 0.8, 0.1]
    
    return {
        'label': label,
        'compound': compound,
        'probs': probs
    }

def run_roberta(text, model, tokenizer, device='cpu'):
    """Run RoBERTa sentiment classification on a single tweet."""
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probs))
    
    return {
        'label': LABELS[predicted_class],
        'probs': probs.tolist(),
        'confidence': float(probs[predicted_class])
    }

def run_llm(text, hf_token, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Run Llama 3 via HuggingFace Inference API as a tiebreaker
    for low confidence predictions.
    
    Only called when RoBERTa and VADER disagree significantly.
    Uses the same few-shot prompt engineering validated in notebooks.
    """
    client = InferenceClient(model=model, token=hf_token)
    
    prompt = f"""You are a sentiment classifier. Classify tweets as positive, negative, or neutral.

Here are some examples:

Tweet: "I love this so much!!!"
{{"label": "positive", "confidence": "1.0", "reasoning": "Strong expression of enthusiasm"}}

Tweet: "This is the worst thing ever."
{{"label": "negative", "confidence": "1.0", "reasoning": "Strongly negative superlative language"}}

Tweet: "I went to the store today."
{{"label": "neutral", "confidence": "0.9", "reasoning": "Factual statement with no emotional language"}}

Now classify this tweet:
Tweet: "{text}"

Respond with only a JSON object in this exact format, nothing else:
{{"label": "positive", "confidence": "0.0", "reasoning": "brief explanation"}}

The label must be exactly one of: positive, negative, neutral.
The confidence score must be a value between 0.0 and 1.0.
Do not include any text outside the JSON object."""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=150,
            temperature=0.0,
        )
        
        cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
        parsed = json.loads(cleaned)
        
        return {
            'label': parsed['label'],
            'confidence': float(parsed['confidence']),
            'reasoning': parsed['reasoning'],
            'parse_error': False,
            'source': 'llm'
        }
    except Exception as e:
        return {
            'label': None,
            'confidence': None,
            'reasoning': str(e),
            'parse_error': True,
            'source': 'llm'
        }

def compute_confidence(roberta_confidence, vader_compound, roberta_label):
    """
    Assess overall prediction confidence based on RoBERTa certainty 
    and VADER agreement.
    
    HIGH   — RoBERTa confident + VADER broadly agrees
    MEDIUM — RoBERTa confident but VADER disagrees  
    LOW    — RoBERTa not confident regardless of VADER
    """
    low_roberta = roberta_confidence < ROBERTA_CONFIDENCE_THRESHOLD
    
    vader_disagrees = (
        (roberta_label == 'negative' and vader_compound > VADER_DISAGREEMENT_THRESHOLD) or
        (roberta_label == 'positive' and vader_compound < -VADER_DISAGREEMENT_THRESHOLD) or
        (roberta_label == 'neutral' and abs(vader_compound) > 0.50)
    )
    
    if low_roberta:
        return 'LOW'
    elif vader_disagrees:
        return 'MEDIUM'
    else:
        return 'HIGH'
        
def compute_triage(roberta_label, roberta_confidence, vader_compound):
    """
    Compute triage priority for brand monitoring.

    VERIFY  — negative + uncertain (low RoBERTa confidence or strong VADER disagreement)
               Ambiguous negative content — human should verify before responding
    RESPOND — negative + confident (high RoBERTa confidence and VADER broadly agrees)
               Clearly negative content — prioritize for brand response  
    MONITOR — neutral or positive
               No immediate action needed — monitor only
    """
    if roberta_label != 'negative':
        return 'MONITOR'

    low_confidence = roberta_confidence < ROBERTA_CONFIDENCE_THRESHOLD
    strong_disagreement = vader_compound > VADER_DISAGREEMENT_THRESHOLD

    if low_confidence or strong_disagreement:
        return 'VERIFY'
    else:
        return 'RESPOND'

def run_ner(text, ner_pipe):
    """Extract named entities using HuggingFace NER.
    MISC excluded."""
    label_map = {'PER': 'PERSON', 'ORG': 'ORG', 'LOC': 'GPE'}
    
    try:
        results = ner_pipe(text)
        entities = [
            {'text': r['word'].strip(), 'type': label_map.get(r['entity_group'], r['entity_group'])}
            for r in results
            if r['entity_group'] in ['PER', 'ORG', 'LOC']
        ]
    except Exception:
        entities = []
    
    return entities

def analyze_tweet(text, models):
    """
    Run the full sentiment analysis pipeline on a single tweet.

    Returns:
        text             — original tweet
        sentiment        — negative / neutral / positive (RoBERTa, or LLM if LOW confidence)
        confidence       — HIGH / MEDIUM / LOW (RoBERTa + VADER agreement)
        triage_priority  — VERIFY / RESPOND / MONITOR (experimental signal)
        entities         — list of {text, type} dicts
        llm_used         — True if Llama 3 was invoked as tiebreaker
        llm_reasoning    — Llama 3 explanation if invoked, else None
        model_breakdown  — per-model predictions, scores, and agreement flag
    """
    # Primary classifier
    roberta_result = run_roberta(
        text,
        models['roberta'],
        models['tokenizer'],
        models['device']
    )

    # Disagreement signal
    vader_result = run_vader(text, models['vader'])

    models_agree = vader_result['label'] == roberta_result['label']

    # Confidence computation
    confidence = compute_confidence(
        roberta_confidence=roberta_result['confidence'],
        vader_compound=vader_result['compound'],
        roberta_label=roberta_result['label']
    )

    llm_result = None
    final_sentiment = roberta_result['label']
    
    if confidence == 'LOW' and models.get('hf_token'):
        llm_result = run_llm(text, models['hf_token'])
        if not llm_result['parse_error'] and llm_result['label']:
            final_sentiment = llm_result['label']

    # Triage, uses RoBERTa confidence + VADER compound
    priority = compute_triage(
        roberta_label=roberta_result['label'],
        roberta_confidence=roberta_result['confidence'],
        vader_compound=vader_result['compound']
    )

    # Named entity extraction
    entities = run_ner(text, models['ner'])

    return {
        'text': text,
        'sentiment': final_sentiment,
        'confidence': confidence,
        'triage_priority': priority,
        'entities': entities,
        'llm_used': llm_result is not None and not llm_result.get('parse_error', True),
        'llm_reasoning': llm_result['reasoning'] if llm_result and not llm_result.get('parse_error') else None,
        'model_breakdown': {
            'roberta_label': roberta_result['label'],
            'roberta_confidence': round(roberta_result['confidence'], 3),
            'vader_label': vader_result['label'],
            'vader_compound': round(vader_result['compound'], 3),
            'models_agree': models_agree,
            'llm_label': llm_result['label'] if llm_result and not llm_result.get('parse_error') else None,
        }
    }
    