# tests/test_pipeline.py
# Lightweight tests for CI/CD — no GPU required, no model loading
# Tests pure logic functions only

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import compute_triage, run_vader, LABELS, ROBERTA_CONFIDENCE_THRESHOLD, VADER_DISAGREEMENT_THRESHOLD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# TRIAGE TESTS
# ============================================================

def test_triage_monitor_for_positive():
    """Positive sentiment should always be MONITOR."""
    assert compute_triage('positive', 0.95, -0.8) == 'MONITOR'

def test_triage_monitor_for_neutral():
    """Neutral sentiment should always be MONITOR."""
    assert compute_triage('neutral', 0.95, 0.0) == 'MONITOR'

def test_triage_respond_high_confidence_agreement():
    """High confidence negative + VADER agrees = RESPOND."""
    assert compute_triage('negative', 0.95, -0.6) == 'RESPOND'

def test_triage_verify_low_confidence():
    """Low RoBERTa confidence on negative = VERIFY."""
    assert compute_triage('negative', 0.50, -0.6) == 'VERIFY'

def test_triage_verify_strong_disagreement():
    """Strong VADER disagreement on negative = VERIFY."""
    assert compute_triage('negative', 0.85, 0.5) == 'VERIFY'

def test_triage_verify_both_signals():
    """Low confidence AND strong disagreement = VERIFY."""
    assert compute_triage('negative', 0.55, 0.4) == 'VERIFY'

def test_triage_boundary_confidence():
    """Exactly at confidence threshold — below = VERIFY."""
    below = ROBERTA_CONFIDENCE_THRESHOLD - 0.01
    above = ROBERTA_CONFIDENCE_THRESHOLD + 0.01
    assert compute_triage('negative', below, -0.6) == 'VERIFY'
    assert compute_triage('negative', above, -0.6) == 'RESPOND'

def test_triage_boundary_disagreement():
    """Exactly at disagreement threshold — above = VERIFY."""
    below = VADER_DISAGREEMENT_THRESHOLD - 0.01
    above = VADER_DISAGREEMENT_THRESHOLD + 0.01
    assert compute_triage('negative', 0.85, below) == 'RESPOND'
    assert compute_triage('negative', 0.85, above) == 'VERIFY'

# ============================================================
# VADER TESTS
# ============================================================

def test_vader_positive():
    """Clearly positive tweet should return positive label."""
    sia = SentimentIntensityAnalyzer()
    result = run_vader("I love this product so much!!!", sia)
    assert result['label'] == 'positive'
    assert result['compound'] > 0.05

def test_vader_negative():
    """Clearly negative tweet should return negative label."""
    sia = SentimentIntensityAnalyzer()
    result = run_vader("This is absolutely terrible, worst experience ever.", sia)
    assert result['label'] == 'negative'
    assert result['compound'] < -0.05

def test_vader_neutral():
    """Factual tweet should return neutral label."""
    sia = SentimentIntensityAnalyzer()
    result = run_vader("The store opens at 9am tomorrow.", sia)
    assert result['label'] == 'neutral'

def test_vader_returns_probs():
    """VADER result should include probability distribution summing to 1."""
    sia = SentimentIntensityAnalyzer()
    result = run_vader("I love this!", sia)
    assert 'probs' in result
    assert len(result['probs']) == 3
    assert abs(sum(result['probs']) - 1.0) < 0.01

def test_vader_compound_range():
    """Compound score should always be between -1 and 1."""
    sia = SentimentIntensityAnalyzer()
    tweets = [
        "I LOVE THIS SO MUCH!!!",
        "I hate everything about this",
        "The weather is okay today"
    ]
    for tweet in tweets:
        result = run_vader(tweet, sia)
        assert -1.0 <= result['compound'] <= 1.0

# ============================================================
# CONSTANTS TESTS
# ============================================================

def test_labels_complete():
    """LABELS should cover all three sentiment classes."""
    assert 0 in LABELS
    assert 1 in LABELS
    assert 2 in LABELS
    assert set(LABELS.values()) == {'negative', 'neutral', 'positive'}

def test_thresholds_reasonable():
    """Thresholds should be in valid ranges."""
    assert 0 < ROBERTA_CONFIDENCE_THRESHOLD < 1
    assert 0 < VADER_DISAGREEMENT_THRESHOLD < 1
