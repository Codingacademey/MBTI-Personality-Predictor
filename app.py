
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# Utilities
# -----------------------------
def ensure_nltk_resources() -> None:
    try:
        import nltk  # noqa: F401
        from nltk.corpus import stopwords  # noqa: F401
        from nltk.stem import WordNetLemmatizer  # noqa: F401
        from nltk.tokenize import word_tokenize  # noqa: F401
    except LookupError:
        import nltk
        # Include both 'punkt' and 'punkt_tab' to satisfy newer NLTK versions
        for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    except Exception:
        # If NLTK is not installed inside the model pipeline environment, we still proceed.
        pass


# Provide the same TextCleaner used during training so pickle can resolve it
class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        try:
            words = nltk.word_tokenize(text)
        except LookupError:
            # Fallback if punkt resources are unavailable
            words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X expected to be a pandas Series or list-like
        try:
            return X.apply(self.clean_text)
        except AttributeError:
            return [self.clean_text(x) for x in X]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Tuple[Any, Optional[Any]]:
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Accept either a bare pipeline, or a dict with both
    if isinstance(obj, dict):
        pipeline = obj.get("pipeline") or obj.get("model") or obj.get("pipeline_model")
        label_encoder = obj.get("label_encoder") or obj.get("encoder")
        return pipeline, label_encoder

    return obj, None


def get_label_mapping(label_encoder: Optional[Any]) -> Optional[np.ndarray]:
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        return label_encoder.classes_
    
    # Fallback: LabelEncoder sorts labels alphabetically, so we need the correct order
    # This is the alphabetical order that sklearn's LabelEncoder uses
    fallback = np.array([
        "ENFJ", "ENFP", "ENTJ", "ENTP",  # 0-3
        "ESFJ", "ESFP", "ESTJ", "ESTP",  # 4-7
        "INFJ", "INFP", "INTJ", "INTP",  # 8-11
        "ISFJ", "ISFP", "ISTJ", "ISTP",  # 12-15
    ])
    return fallback


def predict_personality(pipeline: Any, text: str) -> Dict[str, Any]:
    ensure_nltk_resources()

    if not text or not text.strip():
        return {"error": "Please provide some text."}

    # Predict class id
    y_pred = pipeline.predict([text])
    y_pred = int(y_pred[0])

    # Predict probabilities if available
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])[0]
        except Exception:
            proba = None

    return {"y_pred": y_pred, "proba": proba}


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="MBTI Personality Predictor", page_icon="🧠", layout="centered")

# Premium dark theme with modern styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
      
      .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: auto;
      }
      
      /* Dark theme base */
      .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
      }
      
      /* Header styling */
      .title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        # -webkit-background-clip: text;
        # -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
      }
      
      .subtitle {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
      }
      
      /* Premium prediction card */
      .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid transparent;
        background-clip: padding-box;
        border-radius: 24px;
        padding: 32px;
        margin: 24px 0;
        box-shadow: 
          0 20px 40px rgba(0, 0, 0, 0.3),
          0 0 0 1px rgba(102, 126, 234, 0.2),
          inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        animation: cardGlow 2s ease-in-out infinite alternate;
      }
      
      @keyframes cardGlow {
        0% { box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(102, 126, 234, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1); }
        100% { box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1); }
      }
      
      .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 24px;
        z-index: -1;
      }
      
      .personality-name {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.5);
        margin-bottom: 24px;
        animation: textGlow 3s ease-in-out infinite alternate;
      }
      
      @keyframes textGlow {
        0% { text-shadow: 0 0 40px rgba(102, 126, 234, 0.5); }
        100% { text-shadow: 0 0 60px rgba(102, 126, 234, 0.8), 0 0 80px rgba(118, 75, 162, 0.3); }
      }
      
      /* Trait boxes */
      .trait-container {
        display: flex;
        gap: 16px;
        margin-top: 24px;
      }
      
      .trait-box {
        flex: 1;
        padding: 20px;
        border-radius: 16px;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.5;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }
      
      .trait-box:hover {
        transform: translateY(-2px);
      }
      
      .strength-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
      }
      
      .watchout-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
      }
      
      .trait-label {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 8px;
        display: block;
      }
      
      .strength-label {
        color: #22c55e;
      }
      
      .watchout-label {
        color: #ef4444;
      }
      
      /* Button styling */
      .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        text-transform: none;
        font-size: 1rem;
      }
      
      .stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
      }
      
      .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
      }
      
      .stButton > button:last-child {
        background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        color: #d1d5db;
        box-shadow: 0 4px 15px rgba(55, 65, 81, 0.3);
      }
      
      .stButton > button:last-child:hover {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(55, 65, 81, 0.4);
      }
      
      /* Text area styling */
      .stTextArea > div > div > textarea {
        background: rgba(31, 41, 55, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.5);
        border-radius: 16px;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        padding: 16px;
        transition: all 0.3s ease;
      }
      
      .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: rgba(31, 41, 55, 0.9);
      }
      
      .stTextArea > div > div > textarea::placeholder {
        color: #9ca3af;
      }
      
      /* Progress bar styling */
      .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 6px;
      }
      
      /* Probabilities styling */
      .prob-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      }
      
      .prob-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
        text-align: center;
      }
      
      .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 12px 0;
      }
      
      .prob-label {
        width: 70px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        color: #d1d5db;
      }
      
      .prob-bar {
        flex: 1;
        height: 8px;
        background: rgba(55, 65, 81, 0.5);
        border-radius: 10px;
        overflow: hidden;
      }
      
      .prob-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.8s ease;
      }
      
      .prob-percent {
        width: 50px;
        text-align: right;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        color: #9ca3af;
      }
      
      /* Hide default Streamlit elements */
      .stApp > header { display: none; }
      .stApp > div[data-testid="stToolbar"] { display: none; }
      .stApp > div[data-testid="stDecoration"] { display: none; }
    </style>
    <div class="title">🧠 MBTI Personality Predictor</div>
    <div class="subtitle">Discover your personality type through text analysis</div>
    """,
    unsafe_allow_html=True,
)


# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mbti_model.pkl")
model_loaded = True
try:
    pipeline, label_encoder = load_model(MODEL_PATH)
except FileNotFoundError:
    model_loaded = False
    st.error("Model file `mbti_model.pkl` not found in the project directory.")
    st.stop()
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")
    st.stop()


label_mapping = get_label_mapping(label_encoder)

mbti_traits = {
    "INFP": {
        "good": "You’re kind-hearted, imaginative, and deeply value emotional connections.",
        "bad": "You can be overly idealistic and take things too personally.",
    },
    "INFJ": {
        "good": "You’re insightful, compassionate, and great at understanding others’ emotions.",
        "bad": "You often overthink and struggle with setting boundaries.",
    },
    "ENFP": {
        "good": "You’re energetic, creative, and inspire people with your enthusiasm.",
        "bad": "You may get distracted easily and struggle with consistency.",
    },
    "ENTP": {
        "good": "You’re witty, confident, and love exploring new ideas.",
        "bad": "You can come off as argumentative or dismissive of others’ opinions.",
    },
    "INTP": {
        "good": "You’re analytical, curious, and love exploring deep concepts.",
        "bad": "You can be detached and struggle to express emotions.",
    },
    "INTJ": {
        "good": "You’re strategic, disciplined, and future-focused.",
        "bad": "You can be too perfectionist or impatient with others.",
    },
    "ISFP": {
        "good": "You’re gentle, artistic, and live in the moment.",
        "bad": "You avoid conflict and may struggle with long-term decisions.",
    },
    "ISTP": {
        "good": "You’re practical, cool-headed, and handle problems efficiently.",
        "bad": "You might come off as emotionally distant or impulsive.",
    },
    "ESFP": {
        "good": "You’re fun, spontaneous, and love making people happy.",
        "bad": "You can get bored quickly and dislike responsibilities.",
    },
    "ESTP": {
        "good": "You’re bold, adventurous, and great under pressure.",
        "bad": "You can act recklessly or ignore long-term consequences.",
    },
    "ESFJ": {
        "good": "You’re caring, loyal, and always support your friends.",
        "bad": "You might care too much about others’ opinions.",
    },
    "ESTJ": {
        "good": "You’re dependable, organized, and a natural leader.",
        "bad": "You can be controlling or stubborn at times.",
    },
    "ISFJ": {
        "good": "You’re loyal, thoughtful, and protective of people you love.",
        "bad": "You can be overly selfless and afraid of change.",
    },
    "ISTJ": {
        "good": "You’re responsible, consistent, and detail-oriented.",
        "bad": "You may resist change or new ideas.",
    },
    "ENFJ": {
        "good": "You’re inspiring, empathetic, and deeply value harmony.",
        "bad": "You sometimes neglect your own needs while helping others.",
    },
    "ENTJ": {
        "good": "You’re ambitious, confident, and make things happen.",
        "bad": "You can be too bossy or impatient with others’ pace.",
    },
}


with st.form("predict_form", clear_on_submit=False):
    text_input = st.text_area(
        "Your text",
        height=200,
        placeholder="Write a few paragraphs that reflect your thoughts, preferences, and behaviors...",
    )

    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        submitted = st.form_submit_button("Predict", use_container_width=True)
    with col2:
        reset = st.form_submit_button("Clear", use_container_width=True)


if reset:
    st.experimental_rerun()


user_text = text_input


if submitted:
    if not model_loaded:
        st.stop()

    # Lightweight animation: progress bar then balloons on success
    progress = st.progress(0, text="Analyzing text...")
    for i in range(0, 85, 7):
        progress.progress(i, text="Analyzing text...")
    with st.spinner("Running prediction..."):
        result = predict_personality(pipeline, user_text)

    if "error" in result:
        st.warning(result["error"]) 
    else:
        y_pred = result["y_pred"]
        proba = result.get("proba")

        mbti_label = None
        try:
            if label_mapping is not None and 0 <= y_pred < len(label_mapping):
                mbti_label = str(label_mapping[y_pred])
        except Exception:
            mbti_label = None

        # If we can't map the label id, show the numeric prediction
        if mbti_label is None:
            mbti_label = f"Class {y_pred}"

        # Premium prediction card with animations
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='personality-name'> Result:{mbti_label}</div>", unsafe_allow_html=True)
        
        trait = mbti_traits.get(mbti_label)
        if trait:
            st.markdown(
                f"""
                <div class='trait-container'>
                    <div class='trait-box strength-box'>
                        <span class='trait-label strength-label'>💪 Strength</span>
                        {trait['good']}
                    </div>
                    <div class='trait-box watchout-box'>
                        <span class='trait-label watchout-label'>⚠️ Watch Out</span>
                        {trait['bad']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.balloons()

        if proba is not None and hasattr(np, "argsort") and label_mapping is not None:
            try:
                # Show top-5 probable classes
                top_k = min(5, len(proba))
                top_idx = np.argsort(proba)[-top_k:][::-1]
                labels = [str(label_mapping[i]) if 0 <= i < len(label_mapping) else str(i) for i in top_idx]
                scores = [float(proba[i]) for i in top_idx]

                st.markdown("<div class='prob-card'>", unsafe_allow_html=True)
                st.markdown("<div class='prob-title'>📊 Top Probabilities</div>", unsafe_allow_html=True)
                for label, score in zip(labels, scores):
                    pct = int(round(score * 100))
                    st.markdown(
                        f"<div class='prob-row'>"
                        f"<div class='prob-label'>{label}</div>"
                        f"<div class='prob-bar'><div class='prob-fill' style='width:{pct}%;'></div></div>"
                        f"<div class='prob-percent'>{pct}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                pass


with st.expander("About this app", expanded=False):
    st.markdown(
        "This demo wraps a trained text classification pipeline to predict MBTI types. "
        "Provide enough text for a more reliable prediction."
    )


