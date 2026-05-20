import gradio as gr
import sys
import os
import torch
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_models, analyze_tweet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

HF_TOKEN = os.environ.get("HF_TOKEN")
models = load_models(device=device, hf_token=HF_TOKEN)

FONT = "'IBM Plex Mono', monospace"

EXAMPLES = [
    ["Just received my order from @Nike and the shoes are completely falling apart after one week. Absolutely furious."],
    ["Hey @Apple, any update on when the new MacBook ships? Been waiting 3 weeks."],
    ["Shoutout to @Starbucks for always getting my order right. Love this place!"],
    ["Can't believe @Delta lost my luggage AGAIN. Third time this year. Done with this airline."],
    ["Oh great, @Comcast is down again. What a surprise. Best ISP ever"],
]

SENTIMENT_COLORS = {
    'negative': '#ff4444',
    'neutral':  '#888888',
    'positive': '#44bb44',
}

CONFIDENCE_COLORS = {
    'HIGH':   '#44bb44',
    'MEDIUM': '#ffaa00',
    'LOW':    '#ff4444',
}

CONFIDENCE_DESCRIPTIONS = {
    'HIGH':   'Models agree — high confidence prediction',
    'MEDIUM': 'Models disagree — interpret with caution',
    'LOW':    'Low confidence — prediction may be unreliable',
}

ENTITY_COLORS = {
    'PERSON': '#4a9eff',
    'ORG':    '#ff9f4a',
    'GPE':    '#4aff9f',
}

def badge(text, color):
    return f"""<span style="
        background: {color}22;
        border: 1px solid {color};
        color: {color};
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 12px;
        font-family: {FONT};
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
    ">{text}</span>"""

def section_label(text):
    return f"""<div style="
        color: #555;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: {FONT};
        margin-bottom: 8px;
    ">{text}</div>"""

def card(border_color, content):
    return f"""<div style="
        background: #111122;
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
    ">{content}</div>"""

def analyze(tweet_text):
    if not tweet_text or not tweet_text.strip():
        empty = f"<div style='color:#444; font-family:{FONT}; font-size:13px;'>Submit a tweet to see results.</div>"
        return empty, empty, empty, empty

    result = analyze_tweet(tweet_text.strip(), models)

    sentiment    = result['sentiment']
    confidence   = result['confidence']
    entities     = result['entities']
    breakdown    = result['model_breakdown']
    llm_used     = result.get('llm_used', False)
    llm_reasoning = result.get('llm_reasoning', None)
    triage       = result['triage_priority']

    s_color = SENTIMENT_COLORS[sentiment]
    roberta_pct = int(breakdown['roberta_confidence'] * 100)

    sentiment_html = card(s_color, f"""
        {section_label('Sentiment')}
        <div style="display: flex; align-items: center; gap: 12px;">
            {badge(sentiment.upper(), s_color)}
            <span style="color: #666; font-family: {FONT}; font-size: 13px;">
                RoBERTa — {roberta_pct}% confident
            </span>
        </div>
    """)

    c_color = CONFIDENCE_COLORS[confidence]
    c_desc  = CONFIDENCE_DESCRIPTIONS[confidence]
    agree_text = 'Models agree' if breakdown['models_agree'] else 'Models disagree'
    agree_color = '#44bb44' if breakdown['models_agree'] else '#ff4444'

    llm_section = ""
    if llm_used and llm_reasoning and breakdown.get('llm_label'):
        llm_section = f"""
        <div style="margin-bottom: 16px; border-top: 1px solid #1a1a2e; padding-top: 14px;">
            <div style="color: #a78bfa; font-size: 11px; font-family: {FONT}; margin-bottom: 6px; letter-spacing: 1px;">
                LLAMA 3 — TIEBREAKER (LOW CONFIDENCE)
            </div>
            <div style="color: #ccc; font-size: 14px; font-family: {FONT}; margin-bottom: 6px;">
                {badge(breakdown['llm_label'].upper(), SENTIMENT_COLORS.get(breakdown['llm_label'], '#888'))}
            </div>
            <div style="color: #555; font-size: 12px; font-family: {FONT}; line-height: 1.6;">
                {llm_reasoning}
            </div>
        </div>
        """

    model_html = card('#222', f"""
        {section_label('Model Analysis')}

        <div style="margin-bottom: 16px;">
            <div style="color: #4a9eff; font-size: 11px; font-family: {FONT}; margin-bottom: 6px; letter-spacing: 1px;">
                ROBERTA — PRIMARY CLASSIFIER
            </div>
            <div style="color: #ccc; font-size: 14px; font-family: {FONT}; margin-bottom: 6px;">
                {badge(breakdown['roberta_label'].upper(), s_color)}
                &nbsp;
                <span style="color: #666; font-size: 12px;">{roberta_pct}% confidence</span>
            </div>
            <div style="background: #1a1a2e; border-radius: 4px; height: 5px;">
                <div style="background: #4a9eff; width: {roberta_pct}%; height: 5px; border-radius: 4px;"></div>
            </div>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="color: #ff9f4a; font-size: 11px; font-family: {FONT}; margin-bottom: 6px; letter-spacing: 1px;">
                VADER — SECONDARY SIGNAL
            </div>
            <div style="color: #ccc; font-size: 14px; font-family: {FONT};">
                {badge(breakdown['vader_label'].upper(), SENTIMENT_COLORS[breakdown['vader_label']])}
                &nbsp;
                <span style="color: #666; font-size: 12px;">
                    compound: 
                    <span style="color: {'#ff4444' if breakdown['vader_compound'] < -0.05 else '#44bb44' if breakdown['vader_compound'] > 0.05 else '#888'};">
                        {breakdown['vader_compound']:+.3f}
                    </span>
                </span>
            </div>
        </div>

        {llm_section}

        <div style="border-top: 1px solid #1a1a2e; padding-top: 14px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div>
                    <span style="color: {agree_color}; font-family: {FONT}; font-size: 13px; font-weight: 700;">
                        {agree_text}
                    </span>
                </div>
                <div style="color: #555; font-size: 11px; font-family: {FONT};">·</div>
                <div>
                    <span style="color: {c_color}; font-family: {FONT}; font-size: 13px; font-weight: 700;">
                        {confidence} CONFIDENCE
                    </span>
                </div>
            </div>
            <div style="color: #444; font-size: 11px; font-family: {FONT}; margin-top: 6px;">
                {c_desc}
            </div>
        </div>
    """)

    if entities:
        tags = "".join([
            f"""<span style="
                background: {ENTITY_COLORS.get(e['type'], '#888')}22;
                border: 1px solid {ENTITY_COLORS.get(e['type'], '#888')};
                color: {ENTITY_COLORS.get(e['type'], '#888')};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
                font-family: {FONT};
                margin: 3px 4px 3px 0;
                display: inline-block;
            ">{e['text']} <span style="opacity:0.5; font-size:10px;">{e['type']}</span></span>"""
            for e in entities
        ])
        entity_body = f"<div style='margin-top: 4px;'>{tags}</div>"
        legend = f"""<div style="color: #333; font-size: 10px; font-family: {FONT}; margin-top: 10px;">
            <span style="color:#4a9eff;">■</span> Person &nbsp;
            <span style="color:#ff9f4a;">■</span> Organization &nbsp;
            <span style="color:#4aff9f;">■</span> Location
        </div>"""
    else:
        entity_body = f"<div style='color: #333; font-family: {FONT}; font-size: 13px;'>No named entities detected.</div>"
        legend = ""

    entities_html = card('#1a1a2e', f"""
        {section_label('Named Entities')}
        {entity_body}
        {legend}
    """)

    triage_color = {'VERIFY': '#ff4444', 'RESPOND': '#ffaa00', 'MONITOR': '#44bb44'}.get(triage, '#888')
    triage_desc  = {
        'VERIFY':  'Uncertain negative — verify before acting',
        'RESPOND': 'Confident negative — response recommended',
        'MONITOR': 'Neutral or positive — no action required',
    }.get(triage, '')

    triage_html = f"""
    <div style="background: #0a0a14; border: 1px solid #1a1a2e; border-radius: 8px; padding: 16px 20px; font-family: {FONT};">
        {section_label('Triage Signal')}
        <div style="display: flex; align-items: center; gap: 12px;">
            {badge(triage, triage_color)}
            <span style="color: #444; font-size: 12px;">{triage_desc}</span>
        </div>
        <div style="color: #333; font-size: 10px; margin-top: 10px; line-height: 1.8;">
            RoBERTa confidence threshold: 70% &nbsp;·&nbsp; VADER disagreement threshold: +0.20
        </div>
    </div>
    """

    return sentiment_html, model_html, entities_html, triage_html


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');
body, .gradio-container {{
    background-color: #0a0a14 !important;
    font-family: {FONT} !important;
}}
textarea, input {{
    background: #111122 !important;
    border-color: #222 !important;
    color: #ccc !important;
    font-family: {FONT} !important;
}}
label, .label-wrap {{
    color: #444 !important;
    font-family: {FONT} !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}
button.primary {{
    background: #4a9eff !important;
    border: none !important;
    font-family: {FONT} !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    color: #fff !important;
}}
button.secondary {{
    background: transparent !important;
    border: 1px solid #222 !important;
    color: #444 !important;
    font-family: {FONT} !important;
}}
footer {{ display: none !important; }}
"""

with gr.Blocks(css=CSS) as demo:

    gr.HTML(f"""
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap" rel="stylesheet">
    <div style="text-align: center; padding: 36px 0 28px 0; border-bottom: 1px solid #1a1a2e; margin-bottom: 28px;">
        <div style="
            font-family: {FONT};
            font-size: 10px;
            letter-spacing: 5px;
            color: #4a9eff;
            text-transform: uppercase;
            margin-bottom: 10px;
        ">Multi-Model NLP Pipeline</div>
        <h1 style="
            font-family: {FONT};
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 10px 0;
        ">Tweet Sentiment Analyzer</h1>
        <p style="
            color: #444;
            font-size: 13px;
            font-family: {FONT};
            margin: 0;
            line-height: 1.8;
        ">
            Paste any tweet to analyze its sentiment using RoBERTa and VADER.<br>
            See how each model interprets the text, where they agree, and what entities are mentioned.
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1):
            tweet_input = gr.Textbox(
                label="Tweet",
                placeholder="Paste a tweet here...",
                lines=5,
                max_lines=8,
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze", variant="primary", scale=2)
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)

            gr.Examples(
                examples=EXAMPLES,
                inputs=tweet_input,
                label="Examples",
            )

        with gr.Column(scale=1):
            sentiment_out = gr.HTML()
            model_out     = gr.HTML()
            entities_out  = gr.HTML()
            with gr.Accordion("Triage Signal", open=False):
                gr.HTML(f"""<div style="color: #333; font-size: 11px; font-family: {FONT}; margin-bottom: 10px; line-height: 1.6;">
                    An experimental triage signal derived from RoBERTa confidence and VADER agreement.
                    Thresholds calibrated on the TweetEval sentiment test set (12,284 tweets).
                </div>""")
                triage_out = gr.HTML()

    gr.HTML(f"""
    <div style="
        text-align: center;
        padding: 28px 0 8px 0;
        border-top: 1px solid #1a1a2e;
        margin-top: 24px;
        color: #2a2a3a;
        font-size: 10px;
        font-family: {FONT};
        letter-spacing: 1px;
        line-height: 2;
    ">
        cardiffnlp/twitter-roberta-base-sentiment-latest &nbsp;·&nbsp;
        VADER Sentiment Reasoner &nbsp;·&nbsp;
        dslim/bert-base-NER;
        meta-llama/Meta-Llama-3-8B-Instruct
    </div>
    """)

    submit_btn.click(
        fn=analyze,
        inputs=[tweet_input],
        outputs=[sentiment_out, model_out, entities_out, triage_out]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=[],
        outputs=[tweet_input, sentiment_out, model_out, entities_out, triage_out]
    )

    tweet_input.submit(
        fn=analyze,
        inputs=[tweet_input],
        outputs=[sentiment_out, model_out, entities_out, triage_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
