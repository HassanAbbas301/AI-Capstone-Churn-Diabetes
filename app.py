"""
AI Capstone - one Flask app, two projects:
  * banking    -> customer churn prediction (XGBoost) + Gemini explanation / next action
  * healthcare -> diabetes risk flag (XGBoost) + Gemini explanation (never a diagnosis)

Run:  python app.py   then open http://127.0.0.1:5000
"""
import json
import os
import re
import sqlite3
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:  # the app still runs; factors then come from the fallback below
    xgb = None
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE / "models"))
DB_PATH = os.getenv("DB_PATH", str(BASE / "history.db"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = Flask(__name__)


def norm(name: str) -> str:
    """Collapse the double spaces that exist in the Iranian churn column names."""
    return re.sub(r"\s+", " ", str(name).strip())


def yn(label_hint=""):
    return [[0, "No"], [1, "Yes"]]


# --------------------------------------------------------------------------
# Input schemas (keys are the cleaned training column names)
# --------------------------------------------------------------------------
CHURN_FIELDS = [
    dict(key="Complains", label="Customer has complained", kind="select", options=yn(), default=0, group="Account"),
    dict(key="Status", label="Account status", kind="select", options=[[1, "Active"], [2, "Non-active"]], default=1, group="Account"),
    dict(key="Tariff Plan", label="Tariff plan", kind="select", options=[[1, "Pay as you go"], [2, "Contractual"]], default=1, group="Account"),
    dict(key="Subscription Length", label="Subscription length (months)", kind="number", min=0, max=60, default=30, group="Account"),
    dict(key="Age", label="Customer age", kind="number", min=15, max=90, default=30, group="Account"),
    dict(key="Customer Value", label="Customer value (calculated)", kind="number", min=0, max=3000, default=200, group="Account"),
    dict(key="Charge Amount", label="Charge amount band (0 lowest to 10 highest)", kind="number", min=0, max=10, default=0, group="Billing"),
    dict(key="Call Failure", label="Number of call failures", kind="number", min=0, max=60, default=5, group="Usage"),
    dict(key="Seconds of Use", label="Total seconds of calls", kind="number", min=0, max=20000, default=3000, group="Usage"),
    dict(key="Frequency of use", label="Total number of calls", kind="number", min=0, max=350, default=50, group="Usage"),
    dict(key="Frequency of SMS", label="Total number of SMS", kind="number", min=0, max=700, default=20, group="Usage"),
    dict(key="Distinct Called Numbers", label="Distinct numbers called", kind="number", min=0, max=120, default=20, group="Usage"),
]

AGE_BANDS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
INCOMES = ["< $10k", "$10k-15k", "$15k-20k", "$20k-25k", "$25k-35k", "$35k-50k", "$50k-75k", ">= $75k"]
EDU = ["Never attended / kindergarten", "Grades 1-8", "Grades 9-11", "High school graduate", "Some college", "College graduate"]

DIAB_FIELDS = [
    dict(key="Age", label="Age band", kind="select", options=[[i + 1, b] for i, b in enumerate(AGE_BANDS)], default=8, group="About the person"),
    dict(key="Sex", label="Sex", kind="select", options=[[0, "Female"], [1, "Male"]], default=0, group="About the person"),
    dict(key="Education", label="Education level", kind="select", options=[[i + 1, e] for i, e in enumerate(EDU)], default=5, group="About the person"),
    dict(key="Income", label="Household income", kind="select", options=[[i + 1, e] for i, e in enumerate(INCOMES)], default=5, group="About the person"),
    dict(key="BMI", label="Body mass index (BMI)", kind="number", min=12, max=98, default=27, group="Body and general health", model_cap=60),
    dict(key="GenHlth", label="General health (1 excellent to 5 poor)", kind="select", options=[[1, "1 - Excellent"], [2, "2 - Very good"], [3, "3 - Good"], [4, "4 - Fair"], [5, "5 - Poor"]], default=3, group="Body and general health"),
    dict(key="MentHlth", label="Days of poor mental health (last 30)", kind="number", min=0, max=30, default=0, group="Body and general health"),
    dict(key="PhysHlth", label="Days of poor physical health (last 30)", kind="number", min=0, max=30, default=0, group="Body and general health"),
    dict(key="DiffWalk", label="Difficulty walking or climbing stairs", kind="select", options=yn(), default=0, group="Body and general health"),
    dict(key="HighBP", label="Told they have high blood pressure", kind="select", options=yn(), default=0, group="Reported conditions"),
    dict(key="HighChol", label="Told they have high cholesterol", kind="select", options=yn(), default=0, group="Reported conditions"),
    dict(key="CholCheck", label="Cholesterol checked in last 5 years", kind="select", options=yn(), default=1, group="Reported conditions"),
    dict(key="Stroke", label="Ever had a stroke", kind="select", options=yn(), default=0, group="Reported conditions"),
    dict(key="HeartDiseaseorAttack", label="Heart disease or heart attack", kind="select", options=yn(), default=0, group="Reported conditions"),
    dict(key="Smoker", label="Smoked 100+ cigarettes in lifetime", kind="select", options=yn(), default=0, group="Lifestyle"),
    dict(key="PhysActivity", label="Physical activity in last 30 days", kind="select", options=yn(), default=1, group="Lifestyle"),
    dict(key="Fruits", label="Eats fruit daily", kind="select", options=yn(), default=1, group="Lifestyle"),
    dict(key="Veggies", label="Eats vegetables daily", kind="select", options=yn(), default=1, group="Lifestyle"),
    dict(key="HvyAlcoholConsump", label="Heavy alcohol consumption", kind="select", options=yn(), default=0, group="Lifestyle"),
    dict(key="AnyHealthcare", label="Has health care coverage", kind="select", options=yn(), default=1, group="Access to care"),
    dict(key="NoDocbcCost", label="Skipped a doctor visit because of cost", kind="select", options=yn(), default=0, group="Access to care"),
]

PROJECTS = {
    "banking": dict(
        title="Banking: customer churn",
        entity="customer",
        model_file="xgb_churn_model.pkl",
        cols_file="churn_feature_columns.pkl",
        fields=CHURN_FIELDS,
        levels=[(0.35, "Low"), (0.65, "Medium"), (1.01, "High")],
        flag_true="Flagged: likely to churn",
        flag_false="Not flagged",
        banner="Churn risk score from an XGBoost model trained on Iranian telecom customer data (UCI).",
        healthcare=False,
    ),
    "healthcare": dict(
        title="Healthcare: diabetes risk flag",
        entity="patient",
        model_file="xgb_diabetes_model.pkl",
        cols_file="diabetes_feature_columns.pkl",
        fields=DIAB_FIELDS,
        levels=[(0.35, "Lower"), (0.65, "Moderate"), (1.01, "Elevated")],
        flag_true="Flagged for staff review",
        flag_false="Not flagged",
        banner="A risk flag for nurses and case managers. It is not a diagnosis and not medical advice.",
        healthcare=True,
    ),
}
for _k, _p in PROJECTS.items():
    _p["field_map"] = {f["key"]: f for f in _p["fields"]}

FLAG_THRESHOLD = 0.5
_cache = {}


def load_project(project):
    """Load model + column order once. Raises a readable error if files are missing."""
    if project in _cache:
        return _cache[project]
    cfg = PROJECTS[project]
    mp, cp = MODEL_DIR / cfg["model_file"], MODEL_DIR / cfg["cols_file"]
    if not mp.exists() or not cp.exists():
        raise FileNotFoundError(
            f"Missing model files for '{project}'. Copy {cfg['model_file']} and {cfg['cols_file']} into {MODEL_DIR}."
        )
    model, cols = joblib.load(mp), list(joblib.load(cp))
    unknown = [c for c in cols if norm(c) not in cfg["field_map"]]
    if unknown:
        raise RuntimeError(f"Model columns without a form field: {unknown}. Update the schema in app.py.")
    _cache[project] = (model, cols)
    return _cache[project]


# --------------------------------------------------------------------------
# Database (history for the dashboard)
# --------------------------------------------------------------------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT, created TEXT,
            inputs TEXT, score REAL, flagged INTEGER, level TEXT,
            drivers TEXT, explanation TEXT, source TEXT)"""
    )
    return con


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def validate(project, data):
    cfg, clean = PROJECTS[project], {}
    for f in cfg["fields"]:
        if f["key"] not in data:
            raise ValueError(f"Missing value: {f['label']}")
        try:
            v = float(data[f["key"]])
        except (TypeError, ValueError):
            raise ValueError(f"'{f['label']}' must be a number")
        if f["kind"] == "select":
            if v not in [float(o[0]) for o in f["options"]]:
                raise ValueError(f"'{f['label']}' has an invalid option")
        elif not (f["min"] <= v <= f["max"]):
            raise ValueError(f"'{f['label']}' must be between {f['min']} and {f['max']}")
        clean[f["key"]] = v
    return clean


def describe_value(f, v):
    if f["kind"] == "select":
        return next(str(o[1]) for o in f["options"] if float(o[0]) == v)
    return f"{v:g}"


def predict(project, inputs):
    model, cols = load_project(project)
    cfg = PROJECTS[project]
    row = {}
    for c in cols:
        f = cfg["field_map"][norm(c)]
        v = inputs[f["key"]]
        if f.get("model_cap") is not None:
            v = min(v, f["model_cap"])  # BMI was capped at 60 in training
        row[c] = v
    X = pd.DataFrame([row], columns=cols).astype(float)
    score = float(model.predict_proba(X)[0, 1])

    contribs = None
    try:  # per-feature contributions (log-odds) from the XGBoost booster
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    except Exception as e:
        app.logger.warning("pred_contribs unavailable, using fallback: %s", e)
        # Fallback: how much does the score move if this feature is set to its default value?
        contribs = []
        for c in cols:
            alt = X.copy()
            alt[c] = float(cfg["field_map"][norm(c)]["default"])
            contribs.append(score - float(model.predict_proba(alt)[0, 1]))
    drivers = []
    for c, contrib in zip(cols, contribs):
        f = cfg["field_map"][norm(c)]
        drivers.append(dict(feature=f["label"], value=describe_value(f, inputs[f["key"]]), contribution=float(contrib)))
    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    drivers = [dict(d, direction="raises" if d["contribution"] > 0 else "lowers") for d in drivers[:5]]

    level = next(name for limit, name in cfg["levels"] if score < limit)
    return score, score >= FLAG_THRESHOLD, level, drivers


# --------------------------------------------------------------------------
# Gen AI layer (Gemini) - every prompt is built from the model's output
# --------------------------------------------------------------------------
_client = None


def llm(system, prompt, temperature=0.4):
    """Return model text, or None if there is no key / the call fails."""
    global _client
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types

        if _client is None:
            _client = genai.Client(api_key=key)
        r = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature, max_output_tokens=1000),
        )
        return (r.text or "").strip() or None
    except Exception as e:
        app.logger.warning("Gemini call failed: %s", e)
        return None


def case_context(project, inputs, score, flagged, level, drivers):
    cfg = PROJECTS[project]
    lines = [
        f"MODEL OUTPUT: risk score {score:.2f} (0 to 1), flag threshold {FLAG_THRESHOLD}, "
        f"result = {'FLAGGED' if flagged else 'NOT flagged'}, risk level = {level}.",
        "Note: the model was trained with class re-weighting/balancing, so the score is a ranking-style risk score, "
        "NOT a literal real-world probability. Never say 'X% chance'.",
        "TOP FACTORS (from the model, strongest first):",
    ]
    for d in drivers:
        lines.append(f"- {d['feature']} = {d['value']} -> {d['direction']} the score")
    lines.append("ALL INPUTS:")
    for f in cfg["fields"]:
        lines.append(f"- {f['label']}: {describe_value(f, inputs[f['key']])}")
    return "\n".join(lines)


BANK_SYSTEM = """You assist retention analysts at a bank/telecom. You receive the output of a churn model.
Base every statement on the MODEL OUTPUT, TOP FACTORS and INPUTS provided. Do not invent data.
The wording and the recommended action MUST change with the result: a flagged / High case needs
a retention action, a Low case should explain why the customer looks stable.
Write concise markdown with exactly these sections:
### Why this customer was flagged (or: Why this customer looks stable)
2-4 bullets that reference the actual factors and values.
### Recommended next action
One of: Call to retain / Send targeted offer / Monitor / No action - then one sentence of reasoning.
### Suggested offer copy
2-3 sentences of customer-facing copy (write 'Not needed' if the customer is Low risk).
Do not state the score as a percentage probability."""

HEALTH_SYSTEM = """You are a nurse-style assistant summarising an automated risk flag for hospital staff.
You explain, summarise and inform. HARD RULES (a breach is a failure):
1. NEVER diagnose, and never say or imply the person has, probably has, or does not have diabetes or any condition.
   The result is only a statistical risk flag built from survey-style answers.
2. NEVER tell anyone what to do medically: no treatment, no medication or dose, no diet, exercise or testing instructions.
   Do not use 'you should', 'you must', 'you need to', 'I recommend', 'I advise' or similar.
3. Do not name any medication.
4. Sound like a nurse describing a result to a colleague: calm, plain, neutral. You are not a doctor and never replace one.
5. Say the number is a risk score, not a literal probability, and that a clinician interprets it.
Base everything on the MODEL OUTPUT, TOP FACTORS and INPUTS. The wording must change with the result
(flagged vs not flagged, and which factors mattered). Write concise markdown with exactly these sections:
### What the result shows
2-3 sentences.
### What stood out in the answers
2-4 bullets naming the actual factors and values, and whether they raised or lowered the score.
### General information
2 short bullets of neutral educational context about those factors (what surveys generally associate them with). No advice.
### Questions to raise with a doctor
3 questions the patient could ask, phrased as questions."""

DISCLAIMER = ("_This is an automated risk flag based on survey-style answers. It is not a diagnosis and not medical advice. "
              "Please discuss the result with a doctor._")

UNSAFE = re.compile(
    r"(\byou\s+(have|has|are\s+suffering|suffer|likely\s+have|probably\s+have)\b[^.\n]{0,25}\b(diabet|prediabet))"
    r"|(\b(is|are|being)\s+(a\s+)?diabetic\b)"
    r"|(\bdiagnosed\s+with\b)|(\bthis\s+confirms\b)|(\bconfirmed\s+(diabet|case))"
    r"|(\byou\s+(should|must|need\s+to|have\s+to|ought\s+to)\b)"
    r"|(\bi\s+(recommend|advise|suggest|prescribe)\b)"
    r"|(\b(start|stop|begin|continue)\s+(taking|using|eating|your\s+medication))"
    r"|(\b(metformin|insulin|medication|medicine|prescri\w+|dosage|dose|statin)\b)"
    r"|(\bdon'?t\s+(take|eat|drink)\b)|(\bwill\s+(develop|get)\s+(type\s*2\s+)?(pre)?diabet)"
    r"|(\b(likely|going|bound)\s+to\s+(develop|get|have)\b[^.\n]{0,20}\bdiabet)",
    re.I,
)


def unsafe(text):
    return bool(UNSAFE.search(text or ""))


def template_explanation(project, score, flagged, level, drivers):
    """Deterministic, grounded fallback (no API key / API error / unsafe output)."""
    ups = [d for d in drivers if d["direction"] == "raises"][:3]
    downs = [d for d in drivers if d["direction"] == "lowers"][:2]
    bullet = lambda d: f"- {d['feature']} ({d['value']}) {d['direction']} the score."
    if PROJECTS[project]["healthcare"]:
        head = ("The risk score is above the flag threshold, so this record is flagged for staff review."
                if flagged else "The risk score is below the flag threshold, so this record is not flagged.")
        return (
            f"### What the result shows\n{head} The risk score is {score:.2f} ({level.lower()} level). "
            "It reflects how similar the answers are to patterns seen in survey data. It is not a diagnosis, "
            "and a clinician interprets it.\n\n"
            "### What stood out in the answers\n" + "\n".join(bullet(d) for d in ups + downs) + "\n\n"
            "### General information\n- Surveys generally link several of these factors with higher or lower reported diabetes rates.\n"
            "- The flag only helps staff prioritise who is looked at first.\n\n"
            "### Questions to raise with a doctor\n- How do the factors listed above relate to this result?\n"
            "- What does a risk flag like this mean for me specifically?\n- What information would a doctor want to look at next?"
        )
    action = ("Call to retain" if level == "High" else "Send targeted offer" if level == "Medium" else "Monitor")
    why = "\n".join(bullet(d) for d in ups + downs) or "- Model contributions unavailable."
    offer = "Not needed" if level == "Low" else "We value you as a customer. Reach out and we will review a plan that fits how you use our service."
    return (f"### Why this customer was {'flagged' if flagged else 'not flagged'}\n"
            f"Churn risk score {score:.2f} ({level}).\n{why}\n\n"
            f"### Recommended next action\n**{action}** - based on the {level.lower()} risk level.\n\n"
            f"### Suggested offer copy\n{offer}")


def explain(project, inputs, score, flagged, level, drivers):
    cfg = PROJECTS[project]
    ctx = case_context(project, inputs, score, flagged, level, drivers)
    system = HEALTH_SYSTEM if cfg["healthcare"] else BANK_SYSTEM
    text, source = None, "template"
    for attempt in range(2 if cfg["healthcare"] else 1):
        extra = "" if attempt == 0 else "\nREMINDER: your previous answer broke the hard rules. Do not diagnose, advise, or name medication."
        out = llm(system, ctx + extra)
        if out and not (cfg["healthcare"] and unsafe(out)):
            text, source = out, "gemini"
            break
    if text is None:
        text = template_explanation(project, score, flagged, level, drivers)
    if cfg["healthcare"]:
        text += "\n\n" + DISCLAIMER
    return text, source


CHAT_REFUSAL = ("I can explain what this result shows and which answers influenced it, but I can't diagnose or advise on "
                "treatment, medication, diet or tests. A doctor is the right person for that. Is there a part of the "
                "result or a medical term I can explain?")


def chat_reply(project, case, message, history):
    cfg = PROJECTS[project]
    ctx = case_context(project, case["inputs"], case["score"], bool(case["flagged"]), case["level"], case["drivers"])
    system = (HEALTH_SYSTEM if cfg["healthcare"] else BANK_SYSTEM).split("Write concise markdown")[0]
    system += ("\nYou are now answering follow-up questions about THIS case only. Keep answers under 120 words, "
               "in plain language, grounded in the case data. If asked something unrelated to the case, say so briefly.")
    if cfg["healthcare"]:
        
     system += "\nIf asked for a diagnosis, treatment, medication or what to do medically, decline politely and point to a doctor."
     system += "\nWhen you decline, add one short sentence offering what you can explain about the result."
    convo = "\n".join(f"{'User' if h.get('role') == 'user' else 'Assistant'}: {h.get('text', '')}" for h in history[-8:])
    prompt = f"{ctx}\n\nPREVIOUS EXPLANATION:\n{case['explanation']}\n\nCONVERSATION:\n{convo}\nUser: {message}\nAssistant:"
    out = llm(system, prompt, temperature=0.3)
    if out is None:
        top = case["drivers"][0] if case["drivers"] else None
        base = f"The risk score is {case['score']:.2f} ({case['level']})."
        if top:
            base += f" The biggest factor was {top['feature']} ({top['value']}), which {top['direction']} the score."
        return base + " (Gemini is not connected, so this is a basic answer.)"
    if cfg["healthcare"] and unsafe(out):
        return CHAT_REFUSAL
    return out


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def row_to_case(r):
    return dict(
        id=r["id"], project=r["project"], created=r["created"], inputs=json.loads(r["inputs"]),
        score=r["score"], flagged=bool(r["flagged"]), level=r["level"],
        drivers=json.loads(r["drivers"]), explanation=r["explanation"], source=r["source"],
    )


def check_project(project):
    if project not in PROJECTS:
        raise KeyError("Unknown project")


@app.errorhandler(Exception)
def handle_error(e):
    code = 404 if isinstance(e, KeyError) else 400 if isinstance(e, ValueError) else 500
    if code == 500:
        app.logger.exception(e)
    msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return jsonify(error=msg), code


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(gemini_connected=bool(os.getenv("GEMINI_API_KEY")), gemini_model=GEMINI_MODEL)


@app.get("/api/schema/<project>")
def schema(project):
    check_project(project)
    cfg = PROJECTS[project]
    keep = ("title", "entity", "flag_true", "flag_false", "banner", "healthcare")
    return jsonify({k: cfg[k] for k in keep} | {"fields": cfg["fields"]})


@app.post("/api/predict/<project>")
def predict_route(project):
    check_project(project)
    inputs = validate(project, request.get_json(force=True) or {})
    score, flagged, level, drivers = predict(project, inputs)
    text, source = explain(project, inputs, score, flagged, level, drivers)
    with db() as con:
        cur = con.execute(
            "INSERT INTO cases (project, created, inputs, score, flagged, level, drivers, explanation, source) VALUES (?,?,?,?,?,?,?,?,?)",
            (project, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(inputs), score, int(flagged), level,
             json.dumps(drivers), text, source),
        )
        row = con.execute("SELECT * FROM cases WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(row_to_case(row))


@app.get("/api/case/<int:case_id>")
def case_route(case_id):
    with db() as con:
        row = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not row:
        raise KeyError("Case not found")
    return jsonify(row_to_case(row))


@app.post("/api/chat")
def chat_route():
    body = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        raise ValueError("Type a question first")
    with db() as con:
        row = con.execute("SELECT * FROM cases WHERE id=?", (body.get("case_id"),)).fetchone()
    if not row:
        raise ValueError("Run a prediction first, then ask about it")
    case = row_to_case(row)
    return jsonify(reply=chat_reply(case["project"], case, message[:600], body.get("history") or []))


@app.get("/api/dashboard/<project>")
def dashboard(project):
    check_project(project)
    with db() as con:
        rows = [dict(r) for r in con.execute("SELECT id, created, score, flagged, level FROM cases WHERE project=? ORDER BY id", (project,))]
    total = len(rows)
    levels = [name for _, name in PROJECTS[project]["levels"]]
    return jsonify(
        kpis=dict(
            total=total,
            flagged=sum(r["flagged"] for r in rows),
            flag_rate=round(100 * sum(r["flagged"] for r in rows) / total, 1) if total else 0,
            avg_score=round(float(np.mean([r["score"] for r in rows])), 3) if total else 0,
        ),
        level_counts={l: sum(1 for r in rows if r["level"] == l) for l in levels},
        timeline=[dict(id=r["id"], score=round(r["score"], 3)) for r in rows[-30:]],
        history=[dict(r, score=round(r["score"], 3), flagged=bool(r["flagged"])) for r in rows[-50:][::-1]],
    )


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
