from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import json
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Core pipeline
from app.core.profiler import profile_data
from app.core.analyzer import analyze_training
from app.core.detector import detect_issues

# LLM
from app.llm.prompts import build_prompt
from app.llm.llm_client import call_llm

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ROOT ------------------
@app.get("/")
def home():
    return {"message": "AI ML Debugger Running"}


# ------------------ TRAIN + DEBUG ------------------
@app.post("/train-debug")
def train_debug(file: UploadFile = File(...), model: str = Form(...)):

    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load dataset
    df = pd.read_csv(file_path)

    # Split features & target (last column assumed as target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split (fixed for consistency)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model selection
    if model == "logistic":
        clf = LogisticRegression()
    elif model == "tree":
        clf = DecisionTreeClassifier()
    elif model == "svm":
        clf = SVC()
    else:
        return {"error": "Invalid model"}

    train_acc_list = []
    val_acc_list = []

    # Training loop (simulate epochs)
    for i in range(5):
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))

        # simulate slight overfitting
        train_acc += 0.02 * i

        train_acc_list.append(min(train_acc, 1.0))
        val_acc_list.append(val_acc)

    # Save logs
    logs = {
        "train_acc": train_acc_list,
        "val_acc": val_acc_list
    }

    with open("data/temp_logs.json", "w") as f:
        json.dump(logs, f)

    # Run debugging pipeline
    profile = profile_data(file_path)
    analysis = analyze_training("data/temp_logs.json")
    issues = detect_issues(profile, analysis)

    # LLM explanation
    prompt = build_prompt(profile, analysis, issues)
    explanation = call_llm(prompt)

    return {
        "model": model,
        "analysis": analysis,
        "issues": issues,
        "explanation": explanation
    }


# ------------------ CHAT ------------------
class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    answer = call_llm(query.question)
    return {
        "answer": answer
    }