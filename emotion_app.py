"""
Emotion Analyser — Professional Streamlit UI
Pipeline : CountVectorizer → TfidfTransformer → LogisticRegression
7 emotions : sadness · anger · love · surprise · fear · joy · neutral
Run : streamlit run emotion_app.py
Place bow_vectorizer.pkl, tfidf_transformer.pkl, emotion_model.pkl in same folder.
"""

import streamlit as st
import pickle, numpy as np
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense · Emotion Analyser",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Professional CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Tokens ── */
:root {
  --bg:       #f7f5f0;
  --surface:  #ffffff;
  --line:     rgba(0,0,0,0.09);
  --line2:    rgba(0,0,0,0.05);
  --ink:      #1a1814;
  --ink2:     #6b6760;
  --ink3:     #a09c97;
  --accent:   #2a52be;
  --radius:   12px;
  --radius-sm:8px;
}

/* ── Base ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--ink);
}
[data-testid="stHeader"]          { display:none !important; }
[data-testid="stSidebar"]         { display:none !important; }
#MainMenu, footer                 { visibility:hidden; }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem !important; }

/* ── Wordmark ── */
.wordmark {
  font-family: 'Instrument Serif', serif;
  font-size: 2.6rem;
  color: var(--ink);
  letter-spacing: -.02em;
  text-align: center;
  margin: 0 0 .25rem;
  line-height: 1;
}
.wordmark em { font-style: italic; color: var(--accent); }

.tagline {
  text-align: center;
  font-size: .82rem;
  color: var(--ink3);
  letter-spacing: .12em;
  text-transform: uppercase;
  font-weight: 400;
  margin-bottom: 2.4rem;
}

/* ── Divider ── */
.rule { border: none; border-top: 1px solid var(--line); margin: 1.5rem 0; }

/* ── Section label ── */
.label {
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ink3);
  margin-bottom: .55rem;
}

/* ── Sample chips ── */
div.stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 50px !important;
  color: var(--ink2) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .8rem !important;
  font-weight: 500 !important;
  transition: all .18s ease;
  padding: .45rem 1rem !important;
}
div.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: rgba(42,82,190,0.05) !important;
}

/* ── Textarea ── */
textarea {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important;
  color: var(--ink) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .95rem !important;
  font-weight: 400 !important;
  line-height: 1.65;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
  transition: border .18s;
}
textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(42,82,190,0.1) !important;
}
textarea::placeholder { color: var(--ink3) !important; }

/* ── Primary button ── */
div.stButton > button[kind="primary"] {
  background: var(--ink) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: .88rem !important;
  letter-spacing: .04em;
  padding: .7rem 2rem !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
  transition: all .18s;
}
div.stButton > button[kind="primary"]:hover {
  background: #2a2520 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(0,0,0,0.22) !important;
}
div.stButton > button[kind="primary"]:active { transform: translateY(0); }

/* ── Result card ── */
.result-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1.8rem 2rem;
  margin-top: 1.5rem;
  box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  position: relative;
  overflow: hidden;
}
.result-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--radius) var(--radius) 0 0;
}

/* accent stripe by emotion */
.e-sadness::before  { background: #5b8dd9; }
.e-anger::before    { background: #d94f4f; }
.e-love::before     { background: #d9608a; }
.e-surprise::before { background: #d9a237; }
.e-fear::before     { background: #8860d9; }
.e-joy::before      { background: #4db87a; }
.e-neutral::before  { background: #8c9baa; }

.result-top {
  display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;
}
.e-pill {
  font-size: .68rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; padding: .28rem .8rem;
  border-radius: 50px;
}
.result-title {
  font-family: 'Instrument Serif', serif;
  font-size: 2rem; font-weight: 400;
  line-height: 1; color: var(--ink);
  margin: 0;
}
.result-desc {
  font-size: .9rem; color: var(--ink2); line-height: 1.6; margin-top: .5rem;
}

/* per-emotion pill colours */
.pill-sadness  { background:#eef3fc; color:#2a5499; }
.pill-anger    { background:#fceaea; color:#a33030; }
.pill-love     { background:#fceef3; color:#a3305a; }
.pill-surprise { background:#fdf5e6; color:#9c6c0e; }
.pill-fear     { background:#f2eeff; color:#5e3db3; }
.pill-joy      { background:#eafaf1; color:#1e8a50; }
.pill-neutral  { background:#f0f3f5; color:#4a5a68; }

/* ── Confidence section ── */
.conf-label {
  font-size: .72rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink3); margin: 1.5rem 0 .8rem;
}
.bar-row { display:flex; align-items:center; gap:.7rem; margin-bottom:.55rem; }
.bar-name {
  font-size: .82rem; font-weight: 500; color: var(--ink2);
  min-width: 78px;
}
.bar-track {
  flex: 1; height: 6px; background: rgba(0,0,0,0.07);
  border-radius: 3px; overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 3px; }
.bar-pct { font-size: .75rem; color: var(--ink3); min-width: 36px; text-align: right; font-weight:500; }

/* emotion bar fill colours */
.b-sadness  { background: #5b8dd9; }
.b-anger    { background: #d94f4f; }
.b-love     { background: #d9608a; }
.b-surprise { background: #d9a237; }
.b-fear     { background: #8860d9; }
.b-joy      { background: #4db87a; }
.b-neutral  { background: #8c9baa; }

/* ── Model badge ── */
.model-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 50px; padding: .28rem .8rem;
  font-size: .72rem; color: var(--ink3); font-weight: 500;
  margin-bottom: 2rem;
}
.dot { width:6px; height:6px; border-radius:50%; background:#4db87a; flex-shrink:0; }

/* ── Character counter ── */
.char-count { font-size:.72rem; color:var(--ink3); text-align:right; margin-top:.3rem; }

/* ── Footer ── */
.footer {
  text-align: center; font-size:.72rem; color:var(--ink3);
  margin-top:2.5rem; padding-bottom:1.5rem; line-height:1.8;
}
.footer a { color:var(--ink3); text-decoration:none; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    bow   = pickle.load(open("bow_vectorizer.pkl",  "rb"))
    tfidf = pickle.load(open("tfidf_transformer.pkl","rb"))
    model = pickle.load(open("emotion_model.pkl",    "rb"))
    return bow, tfidf, model

bow_vec, tfidf_tr, clf = load_pipeline()

# ── Emotion config ─────────────────────────────────────────────────────────────
EMOTIONS = {
    0: {
        "key":"sadness",  "label":"Sadness",  "icon":"💧",
        "desc":"The text expresses sorrow, grief, or emotional pain.",
    },
    1: {
        "key":"anger",    "label":"Anger",    "icon":"🔥",
        "desc":"The text conveys frustration, rage, or strong displeasure.",
    },
    2: {
        "key":"love",     "label":"Love",     "icon":"🌸",
        "desc":"The text reflects affection, warmth, tenderness, or deep care.",
    },
    3: {
        "key":"surprise", "label":"Surprise", "icon":"✦",
        "desc":"The text shows astonishment, unexpected discovery, or shock.",
    },
    4: {
        "key":"fear",     "label":"Fear",     "icon":"🌀",
        "desc":"The text expresses anxiety, dread, or apprehension.",
    },
    5: {
        "key":"joy",      "label":"Joy",      "icon":"☀",
        "desc":"The text radiates happiness, enthusiasm, or delight.",
    },
    6: {
        "key":"neutral",  "label":"Neutral",  "icon":"◎",
        "desc":"The text has no strong emotional signal — calm or matter-of-fact.",
    },
}

SAMPLES = {
    "😔 Sadness":   "I've been feeling so empty lately. Nothing excites me anymore and I miss the way things used to be.",
    "😤 Anger":     "I can't believe they cancelled my order without even sending a notification. This is completely unacceptable.",
    "💛 Joy":       "Just got the job offer I've been dreaming about! I'm completely over the moon right now!",
    "🌸 Love":      "Seeing her smile makes everything worthwhile. I've never felt this kind of warmth and belonging before.",
    "😨 Fear":      "I keep thinking something terrible is about to happen. The uncertainty is keeping me up at night.",
    "✦ Surprise":   "I walked in and everyone shouted surprise — I had absolutely no idea they'd planned all this for me!",
}

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="wordmark">Emoti<em>Sense</em></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Natural Language Emotion Analysis</div>', unsafe_allow_html=True)

# model badge
st.markdown("""
<div style="display:flex;justify-content:center;">
  <div class="model-badge">
    <span class="dot"></span>
    LogisticRegression · BoW + TF-IDF · 7 emotions
  </div>
</div>
""", unsafe_allow_html=True)

# sample chips
st.markdown('<div class="label">Try a sample</div>', unsafe_allow_html=True)
chip_cols = st.columns(3)
chosen = None
for col, (lbl, txt) in zip(chip_cols * 2, SAMPLES.items()):
    with col:
        if st.button(lbl, use_container_width=True):
            chosen = txt

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# text input
st.markdown('<div class="label">Your text</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="text",
    value=chosen or "",
    placeholder="Write anything — a thought, a message, a journal entry...",
    height=130,
    label_visibility="collapsed",
    key="main_input",
)

char_count = len(user_text)
st.markdown(f'<div class="char-count">{char_count} characters</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
go = st.button("Analyse Emotion →", type="primary", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if go:
    if not user_text.strip():
        st.warning("Please enter some text to analyse.")
    else:
        bow_feat   = bow_vec.transform([user_text])
        tfidf_feat = tfidf_tr.transform(bow_feat)
        pred_idx   = clf.predict(tfidf_feat)[0]
        proba      = clf.predict_proba(tfidf_feat)[0]

        em  = EMOTIONS[pred_idx]
        key = em["key"]

        # Result card
        st.markdown(f"""
        <div class="result-card e-{key}">
          <div class="result-top">
            <span class="e-pill pill-{key}">{em['icon']} &nbsp;{em['label']}</span>
          </div>
          <div class="result-title">{em['icon']}&ensp;{em['label']}</div>
          <div class="result-desc">{em['desc']}</div>

          <div class="conf-label">Confidence breakdown</div>
          {''.join(f"""
          <div class="bar-row">
            <div class="bar-name">{EMOTIONS[i]['icon']} {EMOTIONS[i]['label']}</div>
            <div class="bar-track"><div class="bar-fill b-{EMOTIONS[i]['key']}" style="width:{p*100:.1f}%"></div></div>
            <div class="bar-pct">{p*100:.1f}%</div>
          </div>""" for i,p in sorted(enumerate(proba), key=lambda x:-x[1]))}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  EmotiSense v1.0 &nbsp;·&nbsp; CountVectorizer (9 482 tokens) · TfidfTransformer · LogisticRegression<br>
  Built with Streamlit
</div>
""", unsafe_allow_html=True)
