# adaptive_translation_app.py
import streamlit as st
from difflib import SequenceMatcher
import time, random, json, csv, io, math, urllib.request, urllib.error, datetime

# ----------------------------
# Optional libraries (safe imports)
# ----------------------------
sacrebleu_available = False
levenshtein_available = False
pandas_available = False
try:
    import sacrebleu
    sacrebleu_available = True
except Exception:
    st.sidebar.info("Optional: sacrebleu not available — BLEU/chrF/TER disabled.")

try:
    import Levenshtein
    levenshtein_available = True
except Exception:
    st.sidebar.info("Optional: python-Levenshtein not available — exact edit-distance disabled.")

try:
    import pandas as pd
    pandas_available = True
except Exception:
    st.sidebar.info("Optional: pandas not available — instructor table will use plain lists.")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Adaptive Translation Tool — Enhanced Stable", layout="wide")
st.title("🌍 Adaptive Translation & Post-Editing Tool — Enhanced (Stable)")

# ----------------------------
# Low-level helpers
# ----------------------------
def safe_sequence_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100.0

def highlight_diff_phrases(student, reference):
    """
    Phrase-level highlighting using token diffs grouped into contiguous phrases.
    Returns HTML string and list of feedback messages.
    """
    matcher = SequenceMatcher(None, reference.split(), student.split())
    highlighted = ""
    feedback = []
    # merge contiguous ops into phrase-level messages
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        stu = " ".join(student.split()[j1:j2])
        ref = " ".join(reference.split()[i1:i2])
        if tag == "equal":
            highlighted += f"<span style='color:green'>{stu} </span>"
        elif tag == "replace":
            highlighted += f"<span style='background:#ffd6d6;color:#a30000;padding:2px;border-radius:3px'>{stu} </span>"
            feedback.append(("replace", stu, ref))
        elif tag == "insert":
            highlighted += f"<span style='background:#fff1cc;color:#7a5200;padding:2px;border-radius:3px'>{stu} </span>"
            feedback.append(("extra", stu, None))
        elif tag == "delete":
            highlighted += f"<span style='background:#d6ecff;color:#004a7a;padding:2px;border-radius:3px'>{ref} </span>"
            feedback.append(("missing", None, ref))
    # Turn feedback tuples into readable messages
    msgs = []
    for t,a,b in feedback:
        if t == "replace":
            msgs.append(f"Replace: '{a}' → '{b}'")
        elif t == "extra":
            msgs.append(f"Extra words: '{a}'")
        elif t == "missing":
            msgs.append(f"Missing: '{b}'")
    return highlighted, msgs

def heuristic_fluency(text):
    """Lightweight, language-agnostic heuristic returning 0-100."""
    if not text or not text.strip():
        return None
    tokens = text.split()
    n = max(1, len(tokens))
    avg_token_len = sum(len(t) for t in tokens)/n
    alpha_ratio = sum(1 for t in tokens if any(c.isalpha() for c in t))/n
    sent_sep = text.count('.') + text.count('?') + text.count('!') + text.count('؟')
    sentences = max(1, sent_sep)
    avg_sent_len = n / sentences
    # heuristics
    score_alpha = alpha_ratio * 40.0
    score_tok = max(0.0, 20.0 - abs(avg_token_len - 5.0)*4.0)
    score_sent = max(0.0, 20.0 - max(0.0, avg_sent_len - 30.0)*0.6)
    punct_density = sum(1 for ch in text if ch in '.,;:!?؟،؛') / (len(text)+1)
    score_punct = max(0.0, 20.0 - punct_density*100.0)
    final = score_alpha + score_tok + score_sent + score_punct
    return max(0.0, min(100.0, final))

# ----------------------------
# Hugging Face Inference API embedding (optional, safe)
# ----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def hf_get_embedding(text: str):
    """
    Calls HF Inference API for sentence-transformer multilingual embedding.
    Requires HF_TOKEN in st.secrets.
    Returns embedding list or None.
    """
    if not text or not text.strip():
        return None
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    url = f"https://api-inference.huggingface.co/models/{model}"
    data = json.dumps({"inputs": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            parsed = json.loads(raw)
            # Expect list or nested list; defensive checks
            if isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], (float, int)):
                return parsed
            if isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], list):
                return parsed[0]
            # If API returns dict with error
            return None
    except Exception:
        return None

def cosine_emb(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return None
    return (dot/(na*nb))*100.0

def semantic_similarity(source_text, student_text, reference_text=None, use_hf=False):
    """
    Returns 0..100 semantic similarity estimate.
    - If use_hf True and HF_TOKEN exists -> use HF embeddings on (source_text or reference_text) vs student_text.
    - Else fallback: simple difflib similarity between reference_text and student_text (if reference given).
    """
    if use_hf:
        # prefer source_text; fallback to reference_text
        left = source_text.strip() if source_text and source_text.strip() else (reference_text or "")
        emb1 = hf_get_embedding(left)
        emb2 = hf_get_embedding(student_text)
        if emb1 and emb2:
            val = cosine_emb(emb1, emb2)
            if val is not None:
                return val
        # HF failed, automatically fall through to fallback
    # fallback surface-level
    if reference_text and reference_text.strip():
        return safe_sequence_ratio(reference_text, student_text)
    return None

# ----------------------------
# Session-state initialization & gamification structures
# ----------------------------
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}          # user -> total points
if "history" not in st.session_state:
    st.session_state.history = []              # list of dicts with recent attempts (cap)
if "streaks" not in st.session_state:
    st.session_state.streaks = {}              # user -> (streak_count, last_date_str)
if "badges" not in st.session_state:
    st.session_state.badges = {}               # user -> set(badges)

# helper to record score event (keeps history bounded)
def record_attempt(user, sem, flu, combined, points):
    now = time.time()
    entry = {"user": user, "time": now, "sem": sem, "flu": flu, "combined": combined, "points": points}
    st.session_state.history.append(entry)
    # cap history size
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]

# badge awarding
def award_badges(user):
    s = st.session_state.leaderboard.get(user, 0)
    if user not in st.session_state.badges:
        st.session_state.badges[user] = set()
    if s >= 100 and "Centurion" not in st.session_state.badges[user]:
        st.session_state.badges[user].add("Centurion")
    if s >= 50 and "Consistent" not in st.session_state.badges[user]:
        st.session_state.badges[user].add("Consistent")
    # streak badge
    streak = st.session_state.streaks.get(user, (0,""))[0]
    if streak >= 5 and "Streaker" not in st.session_state.badges[user]:
        st.session_state.badges[user].add("Streaker")

# ----------------------------
# Small in-app exercise bank
# ----------------------------
EXERCISES = {
    "idioms": [
        ("It's raining cats and dogs.", "تتساقط أمطار غزيرة"),
        ("Break a leg!", "حظاً سعيداً!")],
    "conciseness": [
        ("With light wings of love I climbed these walls; stone boundaries cannot withstand love.",
         "تسلّقت هذه الجدران بأجنحة الحب؛ فالحدود الحجرية لا تصمد أمام الحب.")],
    "grammar": [
        ("She has been to Paris.", "لقد زارت باريس."),
        ("They will have finished by Monday.", "سينتهون بحلول يوم الإثنين.")]
}

# ----------------------------
# UI: Sidebar options
# ----------------------------
st.sidebar.header("Options & Safety")
use_hf = st.sidebar.checkbox("Enable advanced semantic scoring (Hugging Face Inference API)", value=False)
if use_hf:
    st.sidebar.write("Requires `HF_TOKEN` in Streamlit Secrets (Settings → Secrets). HF calls are cached and timeout-protected.")
# allow teacher to turn off BLEU etc if sacrebleu not installed
show_metrics = st.sidebar.checkbox("Show BLEU/chrF/TER (if available)", value=sacrebleu_available)

# username & tabs
username = st.text_input("Your name (used for leaderboard):")

tabs = st.tabs(["Translate & Post-Edit", "Challenges", "Exercises & Badges", "Leaderboard", "Instructor"])

# ----------------------------
# Tab 1: Translate & Post-Edit
# ----------------------------
with tabs[0]:
    st.header("Translate & Post-Edit")
    source = st.text_area("Source text (original)", placeholder="Paste the source (original) text here", height=80)
    reference = st.text_area("Reference translation (human) — optional but recommended", placeholder="Paste human reference translation (if available)", height=80)
    student = st.text_area("Student translation (or MT output to post-edit)", placeholder="Enter your translation here", height=160)

    if st.button("Evaluate translation (safe)"):
        if not username:
            st.warning("Please enter your name before evaluating.")
        elif not student.strip():
            st.warning("Please provide the student translation.")
        else:
            t0 = time.time()
            # phrase-level highlighting if reference present
            if reference and reference.strip():
                html, msgs = highlight_diff_phrases(student, reference)
                st.markdown("**Highlights (compared to reference):**")
                st.markdown(html, unsafe_allow_html=True)
                st.subheader("Word/Phrase feedback:")
                for m in msgs:
                    st.warning(m)
            else:
                st.info("No reference provided — highlights unavailable. Provide a human reference for detailed diagnostics.")
                msgs = []

            # semantic (HF or fallback)
            sem = semantic_similarity(source, student, reference_text=reference, use_hf=use_hf)
            if sem is None:
                st.info("Semantic adequacy unavailable (no reference or HF disabled).")
            else:
                st.metric("Semantic adequacy (0-100)", f"{sem:.2f}")

            # fluency heuristic
            flu = heuristic_fluency(student)
            if flu is not None:
                st.metric("Fluency heuristic (0-100)", f"{flu:.2f}")
            else:
                st.info("Fluency heuristic unavailable.")

            # sacrebleu metrics if requested & available
            if show_metrics and sacrebleu_available and reference and reference.strip():
                try:
                    bleu = sacrebleu.corpus_bleu([student], [[reference]]).score
                    chrf = sacrebleu.corpus_chrf([student], [[reference]]).score
                    ter = sacrebleu.corpus_ter([student], [[reference]]).score
                    st.write(f"BLEU: {bleu:.2f} | chrF: {chrf:.2f} | TER: {ter:.2f}")
                except Exception:
                    st.warning("BLEU/chrF/TER calculation failed (sacrebleu exception).")

            # combined score & points
            combined = None
            if sem is not None and flu is not None:
                combined = 0.65 * sem + 0.35 * flu
                st.metric("Combined Score", f"{combined:.2f}")
            elif sem is not None:
                combined = sem
            elif flu is not None:
                combined = flu

            points = 5 + int((combined or 50) / 20)  # gentle scaling
            st.success(f"Points awarded: {points}")

            # streak logic
            today = datetime.date.today().isoformat()
            streak_info = st.session_state.streaks.get(username, (0,""))
            streak_count, last_date = streak_info
            if last_date == today:
                # same day, increment only if they had no record yet today
                pass
            else:
                # new day activity increments streak (simple logic)
                if last_date:
                    # if yesterday was last_date, increment; else reset
                    yest = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
                    if last_date == yest:
                        streak_count += 1
                    else:
                        streak_count = 1
                else:
                    streak_count = 1
                st.session_state.streaks[username] = (streak_count, today)

            # update leaderboard
            st.session_state.leaderboard[username] = st.session_state.leaderboard.get(username, 0) + points
            award_badges(username)
            record_attempt(username, sem, flu, combined, points)
            # keep history small (cap)
            if len(st.session_state.history) > 200:
                st.session_state.history = st.session_state.history[-200:]

            t1 = time.time()
            st.write(f"Evaluation time: {t1 - t0:.2f} s")

# ----------------------------
# Tab 2: Challenges (timer)
# ----------------------------
with tabs[1]:
    st.header("Timer challenges")
    if st.button("Start challenge"):
        pool = [("I love you.", "أنا أحبك."), ("Knowledge is power.", "المعرفة قوة."), ("The weather is nice today.", "الطقس جميل اليوم.")]
        ch = random.choice(pool)
        st.session_state.challenge = {"source": ch[0], "reference": ch[1], "start": time.time()}
        st.success(f"Challenge: Translate — {ch[0]}")
    if "challenge" in st.session_state:
        st.write("Source:", st.session_state.challenge["source"])
        ans = st.text_area("Your translation (challenge)", key="challenge_ans")
        if st.button("Submit challenge"):
            if not username:
                st.warning("Please enter your name.")
            else:
                elapsed = time.time() - st.session_state.challenge["start"]
                html, msgs = highlight_diff_phrases(ans, st.session_state.challenge["reference"])
                st.markdown(html, unsafe_allow_html=True)
                for m in msgs:
                    st.warning(m)
                # simple scoring based on errors and time
                error_count = len(msgs)
                base = max(1, 15 - error_count)
                time_bonus = max(0, int(10 - elapsed))
                pts = base + time_bonus
                st.success(f"Points: {pts} (errors: {error_count}, time bonus: {time_bonus})")
                st.session_state.leaderboard[username] = st.session_state.leaderboard.get(username,0) + pts
                award_badges(username)
                record_attempt(username, None, None, None, pts)
                # clear challenge to avoid duplicates
                del st.session_state["challenge"]

# ----------------------------
# Tab 3: Exercises & Badges
# ----------------------------
with tabs[2]:
    st.header("Exercises & Badges")
    st.write("Personal streak:", st.session_state.streaks.get(username, (0,""))[0] if username else "—")
    st.write("Badges:", ", ".join(sorted(st.session_state.badges.get(username, set()))) if username else "—")

    # Recommend exercise based on recent feedback (fallback: random)
    rec = None
    if st.session_state.history:
        last = st.session_state.history[-1]
        # simple rule: if combined small or sem missing -> recommend idiom/conciseness
        if last.get("combined") and last["combined"] < 50:
            rec = ("conciseness", random.choice(EXERCISES["conciseness"]))
        else:
            rec = ("idioms", random.choice(EXERCISES["idioms"]))
    else:
        # just offer a random exercise
        cat = random.choice(list(EXERCISES.keys()))
        rec = (cat, random.choice(EXERCISES[cat]))
    st.subheader(f"Recommended exercise — category: {rec[0]}")
    st.write("Source:", rec[1][0])
    st.write("Target example:", rec[1][1])
    attempt = st.text_area("Try your translation (exercise)", key="exercise_attempt")
    if st.button("Submit exercise"):
        if not username:
            st.warning("Enter your name.")
        else:
            # quick feedback via difflib vs example
            sim = safe_sequence_ratio(attempt, rec[1][1])
            st.write(f"Similarity to example: {sim:.2f}%")
            pts = 2 + int(sim/25)
            st.session_state.leaderboard[username] = st.session_state.leaderboard.get(username,0) + pts
            award_badges(username)
            record_attempt(username, None, None, None, pts)
            st.success(f"Exercise points: {pts}")

# ----------------------------
# Tab 4: Leaderboard
# ----------------------------
with tabs[3]:
    st.header("Leaderboard")
    view = st.selectbox("View", ["All time", "Today"], key="leaderboard_view")
    # build records list
    if st.session_state.leaderboard:
        # If 'Today' filter requested, inspect history for today's users
        if view == "All time":
            sorted_lb = sorted(st.session_state.leaderboard.items(), key=lambda x: x[1], reverse=True)
        else:
            today_iso = datetime.date.today().isoformat()
            # calculate points for today from history
            points_today = {}
            for rec in st.session_state.history:
                dt = datetime.datetime.fromtimestamp(rec["time"]).date().isoformat()
                if dt == today_iso:
                    points_today[rec["user"]] = points_today.get(rec["user"],0) + (rec["points"] or 0)
            sorted_lb = sorted(points_today.items(), key=lambda x: x[1], reverse=True)
        for i,(u,p) in enumerate(sorted_lb, start=1):
            st.write(f"{i}. **{u}** — {p} pts")
    else:
        st.info("No leaderboard entries yet.")

# ----------------------------
# Tab 5: Instructor dashboard & export
# ----------------------------
with tabs[4]:
    st.header("Instructor tools")
    # quick stats
    total_users = len(st.session_state.leaderboard)
    total_attempts = len(st.session_state.history)
    st.write(f"Users: {total_users} | Attempts recorded: {total_attempts}")
    # aggregate metrics
    sem_vals = [h["sem"] for h in st.session_state.history if h.get("sem") is not None]
    flu_vals = [h["flu"] for h in st.session_state.history if h.get("flu") is not None]
    combined_vals = [h["combined"] for h in st.session_state.history if h.get("combined") is not None]
    if sem_vals:
        st.write("Class average semantic adequacy:", sum(sem_vals)/len(sem_vals))
    if flu_vals:
        st.write("Class average fluency heuristic:", sum(flu_vals)/len(flu_vals))
    if combined_vals:
        st.write("Class average combined:", sum(combined_vals)/len(combined_vals))

    # show top common feedback messages aggregated (from last 100)
    flat_msgs = []
    # we can also extract messages from highlight diffs stored in history? We saved only numeric metrics; we can reconstruct from last n attempts if reference stored elsewhere
    st.write("Recent attempts (capped):")
    for rec in reversed(st.session_state.history[-50:]):
        t = datetime.datetime.fromtimestamp(rec["time"]).strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"{t} | {rec['user']} | pts: {rec['points']} | sem: {rec['sem']} | flu: {rec['flu']}")

    # export csv of history
    if st.button("Export attempts CSV"):
        # build CSV in-memory
        si = io.StringIO()
        writer = csv.writer(si)
        writer.writerow(["time","user","points","sem","flu","combined"])
        for rec in st.session_state.history:
            writer.writerow([datetime.datetime.fromtimestamp(rec["time"]).isoformat(), rec["user"], rec["points"], rec.get("sem"), rec.get("flu"), rec.get("combined")])
        st.download_button("Download CSV", si.getvalue(), "attempts.csv", "text/csv")

    st.caption("Note: For full semantic scoring using Hugging Face, enable Advanced Mode and set HF_TOKEN in app Secrets. HF calls are cached and will not crash the app if unavailable — the app will fallback automatically.")

# End of app
