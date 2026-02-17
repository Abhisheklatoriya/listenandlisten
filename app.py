import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="French Listening Trainer", page_icon="🎧", layout="centered")

# ----------------------------
# Secrets / env
# ----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MODEL_GRADER = st.secrets.get("OPENAI_MODEL_GRADER", "gpt-4o-mini")
TTS_MODEL = st.secrets.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = st.secrets.get("OPENAI_TTS_VOICE", "alloy")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Item:
    id: str
    level: str
    topic: str
    fr: str
    en: str
    vocab: List[Tuple[str, str]]  # (fr_word, en_word)

# Starter content (expand later)
LIBRARY: List[Item] = [
    Item("a1_001", "A1", "daily", "Bonjour, vous allez bien ?", "Hello, are you doing well?", [("bonjour", "hello"), ("aller", "to go / to be (doing)")]),
    Item("a1_002", "A1", "daily", "Je voudrais un café, s’il vous plaît.", "I would like a coffee, please.", [("voudrais", "would like"), ("s’il vous plaît", "please")]),
    Item("a1_003", "A1", "travel", "Où est la station de métro ?", "Where is the metro station?", [("où", "where"), ("station", "station"), ("métro", "subway")]),
    Item("a2_001", "A2", "work", "Je peux vous envoyer le document cet après-midi.", "I can send you the document this afternoon.", [("envoyer", "to send"), ("document", "document"), ("cet après-midi", "this afternoon")]),
    Item("a2_002", "A2", "daily", "Je n’ai pas compris, pouvez-vous répéter ?", "I didn’t understand, can you repeat?", [("compris", "understood"), ("répéter", "repeat")]),
    Item("b1_001", "B1", "work", "On a dû reporter la réunion à cause d’un imprévu.", "We had to postpone the meeting because of an unexpected issue.", [("reporter", "postpone"), ("réunion", "meeting"), ("imprévu", "unexpected issue")]),
    Item("b1_002", "B1", "daily", "Je vais vérifier et je reviens vers vous dès que possible.", "I’ll check and get back to you as soon as possible.", [("vérifier", "check"), ("revenir vers", "get back to"), ("dès que possible", "as soon as possible")]),
]

LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1"]

def level_index(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0

# ----------------------------
# Spaced repetition (in-memory MVP)
# progress[item_id] = {"due": epoch_seconds, "ease": float, "reps": int, "interval_days": int, "last_score": float}
# ----------------------------
def now_ts() -> int:
    return int(datetime.utcnow().timestamp())

def sr_update(prev: Optional[dict], score: float) -> dict:
    """
    Simplified SM-2-ish:
    - score >= 0.75 => increase reps, ease slightly up, interval grows
    - else => reps reset, due in ~30 minutes, ease down
    """
    is_correct = score >= 0.75

    ease = float(prev["ease"]) if prev else 2.5
    reps = int(prev["reps"]) if prev else 0
    interval = int(prev["interval_days"]) if prev else 0

    if is_correct:
        reps += 1
        ease = min(2.7, ease + 0.05)

        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 3
        else:
            interval = max(3, round(max(1, interval) * ease))

        due_dt = datetime.utcnow() + timedelta(days=interval)
    else:
        reps = 0
        ease = max(1.3, ease - 0.2)
        interval = 0
        due_dt = datetime.utcnow() + timedelta(minutes=30)

    return {
        "due": int(due_dt.timestamp()),
        "ease": float(ease),
        "reps": int(reps),
        "interval_days": int(interval),
        "last_score": float(score),
    }

def pick_due_item(progress: Dict[str, dict]) -> Optional[Item]:
    due_ids = [iid for iid, p in progress.items() if p.get("due", 0) <= now_ts()]
    if not due_ids:
        return None
    due_ids.sort(key=lambda iid: progress[iid]["due"])
    due_id = due_ids[0]
    return next((x for x in LIBRARY if x.id == due_id), None)

def pick_new_item(progress: Dict[str, dict], target_level: str) -> Optional[Item]:
    candidates = [x for x in LIBRARY if x.level == target_level and x.id not in progress]
    if candidates:
        return candidates[0]
    candidates = [x for x in LIBRARY if x.level == target_level]
    return candidates[0] if candidates else None

# ----------------------------
# OpenAI grading + TTS
# ----------------------------
def grade_translation(fr: str, reference_en: str, user_en: str) -> dict:
    """
    Returns strict JSON:
    {"score":0..1,"is_correct":bool,"missing_points":[...],"accepted_paraphrase":str}
    """
    if not client:
        return {
            "score": 0.0,
            "is_correct": False,
            "missing_points": ["No OPENAI_API_KEY set in Streamlit Secrets."],
            "accepted_paraphrase": ""
        }

    system = (
        "You grade English translations of French excerpts.\n"
        "Return strict JSON only:\n"
        '{"score":0..1,"is_correct":boolean,"missing_points":[...],"accepted_paraphrase":string}\n'
        "Grade based on semantic meaning coverage, not literal translation."
    )

    user = f"""French:
{fr}

Reference meaning:
{reference_en}

User meaning:
{user_en}

Return JSON only."""
    resp = client.chat.completions.create(
        model=MODEL_GRADER,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )

    text = resp.choices[0].message.content or "{}"

    import json
    try:
        out = json.loads(text)
    except Exception:
        out = {
            "score": 0.0,
            "is_correct": False,
            "missing_points": ["Could not parse grader JSON output."],
            "accepted_paraphrase": "",
        }

    # normalize
    score = float(out.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    is_correct = bool(out.get("is_correct", score >= 0.75))
    missing_points = out.get("missing_points", [])
    accepted = out.get("accepted_paraphrase", "")

    return {
        "score": score,
        "is_correct": is_correct,
        "missing_points": missing_points if isinstance(missing_points, list) else [str(missing_points)],
        "accepted_paraphrase": str(accepted) if accepted is not None else "",
    }

def tts_audio_bytes(text_fr: str) -> Optional[bytes]:
    """
    Generates audio bytes via OpenAI TTS and caches in session_state.
    Works across common OpenAI Python SDK return shapes.
    """
    if not client:
        return None

    cache = st.session_state.audio_cache
    if text_fr in cache:
        return cache[text_fr]

    try:
        # NOTE: many SDK versions expect `response_format` not `format`
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text_fr,
            response_format="mp3",
        )

        audio_bytes: Optional[bytes] = None

        # Common shapes:
        # - resp.content -> bytes
        # - resp.read() -> bytes
        if hasattr(resp, "content") and isinstance(resp.content, (bytes, bytearray)):
            audio_bytes = bytes(resp.content)
        elif hasattr(resp, "read"):
            audio_bytes = resp.read()
        else:
            # Last resort attempt
            try:
                audio_bytes = bytes(resp)  # type: ignore
            except Exception:
                audio_bytes = None

        if not audio_bytes:
            raise TypeError("TTS response did not contain audio bytes in a supported shape.")

        cache[text_fr] = audio_bytes
        return audio_bytes

    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# ----------------------------
# Session init
# ----------------------------
if "progress" not in st.session_state:
    st.session_state.progress = {}
if "target_level" not in st.session_state:
    st.session_state.target_level = "A1"
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "current" not in st.session_state:
    st.session_state.current = None
if "mode" not in st.session_state:
    st.session_state.mode = "new"
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "show_transcript" not in st.session_state:
    st.session_state.show_transcript = False

def load_next():
    progress = st.session_state.progress
    due = pick_due_item(progress)
    if due:
        st.session_state.current = due
        st.session_state.mode = "review"
        st.session_state.last_feedback = None
        return

    item = pick_new_item(progress, st.session_state.target_level)
    st.session_state.current = item
    st.session_state.mode = "new"
    st.session_state.last_feedback = None

if st.session_state.current is None:
    load_next()

# ----------------------------
# UI
# ----------------------------
st.title("🎧 French Listening Trainer (MVP)")
st.caption("Listen → type the meaning in English → get graded → missed items come back later.")

if not OPENAI_API_KEY:
    st.info("Add OPENAI_API_KEY in Streamlit Secrets to enable audio + grading. The app will still load.")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    st.metric("Level", st.session_state.target_level)
with colB:
    st.metric("Streak", st.session_state.streak)
with colC:
    due_count = sum(1 for _, p in st.session_state.progress.items() if p.get("due", 0) <= now_ts())
    st.metric("Due now", due_count)

item: Optional[Item] = st.session_state.current
if not item:
    st.error("No content available. Add more items to LIBRARY.")
    st.stop()

st.subheader(f"Mode: {'🔁 Review' if st.session_state.mode=='review' else '🆕 New'} • Topic: {item.topic} • Item: {item.id}")

# Audio
audio_bytes = None
if OPENAI_API_KEY:
    with st.spinner("Preparing audio..."):
        audio_bytes = tts_audio_bytes(item.fr)

if audio_bytes:
    st.audio(audio_bytes, format="audio/mp3")
else:
    st.info("Audio unavailable (TTS failed or missing API key). You can still practice using the transcript toggle.")

st.checkbox("Show French transcript", key="show_transcript")
if st.session_state.show_transcript:
    st.write(f"**French:** {item.fr}")

user_answer = st.text_area(
    "Type the meaning in English:",
    placeholder="e.g., 'Where is the metro station?'",
    height=110
)

c1, c2 = st.columns([1, 1])
with c1:
    submitted = st.button("Submit", type="primary", use_container_width=True)
with c2:
    next_clicked = st.button("Next", use_container_width=True)

if next_clicked:
    load_next()
    st.rerun()

if submitted:
    if not user_answer.strip():
        st.warning("Type an English meaning first.")
    else:
        with st.spinner("Grading..."):
            fb = grade_translation(item.fr, item.en, user_answer.strip())

        st.session_state.last_feedback = fb

        # update SR progress
        prev = st.session_state.progress.get(item.id)
        st.session_state.progress[item.id] = sr_update(prev, fb["score"])

        # update streak + level
        if fb["is_correct"]:
            st.session_state.streak += 1
            if st.session_state.streak >= 4:
                idx = min(level_index(st.session_state.target_level) + 1, len(LEVEL_ORDER) - 1)
                st.session_state.target_level = LEVEL_ORDER[idx]
                st.session_state.streak = 0
        else:
            st.session_state.streak = 0
            if level_index(st.session_state.target_level) > 0:
                st.session_state.target_level = LEVEL_ORDER[level_index(st.session_state.target_level) - 1]

        st.divider()
        st.write(f"### Result: {'✅ Correct' if fb['is_correct'] else '❌ Not quite'}")
        st.write(f"**Score:** {fb['score']:.2f}")
        st.write(f"**Correct meaning:** {item.en}")

        if fb.get("accepted_paraphrase"):
            st.write(f"**Accepted paraphrase:** {fb['accepted_paraphrase']}")

        if fb.get("missing_points"):
            st.write("**What was missing / off:**")
            for p in fb["missing_points"]:
                st.write(f"- {p}")

        st.write("**Key vocab:**")
        for frw, enw in item.vocab:
            st.write(f"- **{frw}** = {enw}")

        if fb["is_correct"]:
            st.success("Nice! This item will show up later after a longer interval.")
        else:
            st.info("This item is scheduled to reappear later (about ~30 minutes in this MVP).")

        st.button("Load next item", on_click=load_next)

with st.expander("Progress (debug)"):
    st.json(st.session_state.progress)
