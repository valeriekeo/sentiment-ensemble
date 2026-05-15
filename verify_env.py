"""
verify_env.py — Project 1: Multi-Model Sentiment Ensemble
Run with: python verify_env.py
Checks that all required packages are installed and accessible,
and verifies GPU availability.
"""

import sys
import importlib

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}\n")

# (package_to_import, pip_name, required_version_attr)
PACKAGES = [
    ("torch",           "torch",            "__version__"),
    ("transformers",    "transformers",     "__version__"),
    ("datasets",        "datasets",         "__version__"),
    ("huggingface_hub", "huggingface-hub",  "__version__"),
    ("nltk",            "nltk",             "__version__"),
    ("spacy",           "spacy",            "__version__"),
    ("vaderSentiment",  "vaderSentiment",   "__version__"),
    ("gradio",          "gradio",           "__version__"),
    ("pandas",          "pandas",           "__version__"),
    ("matplotlib",      "matplotlib",       "__version__"),
    ("seaborn",         "seaborn",          "__version__"),
    ("sklearn",         "scikit-learn",     "__version__"),
    ("jupyter",         "jupyter",          "__version__"),
    ("ipykernel",       "ipykernel",        "__version__"),
]

all_good = True

print("=" * 55)
print(f"{'Package':<20} {'Status':<12} {'Version'}")
print("=" * 55)

for import_name, pip_name, version_attr in PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, version_attr, "unknown")
        print(f"{pip_name:<20} {'OK':<12} {version}")
    except ImportError:
        print(f"{pip_name:<20} {'MISSING':<12} — install with: python -m pip install {pip_name}")
        all_good = False

# spaCy model check
print("-" * 55)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"{'en_core_web_sm':<20} {'OK':<12} spaCy model loaded")
except OSError:
    print(f"{'en_core_web_sm':<20} {'MISSING':<12} — install with: python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")
    all_good = False

# NLTK VADER lexicon check
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.polarity_scores("test")
    print(f"{'vader_lexicon':<20} {'OK':<12} NLTK VADER lexicon loaded")
except Exception:
    print(f"{'vader_lexicon':<20} {'MISSING':<12} — run: python -c \"import nltk; nltk.download('vader_lexicon')\"")
    all_good = False

# GPU check
print("-" * 55)
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{'CUDA':<20} {'OK':<12} {name} ({vram:.1f} GB VRAM)")
    else:
        print(f"{'CUDA':<20} {'WARNING':<12} GPU not available — CPU only")
        all_good = False
except Exception as e:
    print(f"{'CUDA':<20} {'ERROR':<12} {e}")
    all_good = False

print("=" * 55)
if all_good:
    print("\nAll checks passed. Environment is ready.")
else:
    print("\nSome checks failed. Fix the issues above and re-run this script.")
