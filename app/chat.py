from app.core.profiler import profile_data
from app.core.analyzer import analyze_training
from app.core.detector import detect_issues
from app.llm.llm_client import call_llm

def ask_question(question):
    profile = profile_data("data/sample_dataset.csv")
    analysis = analyze_training("data/sample_logs.json")
    issues = detect_issues(profile, analysis)

    prompt = f"""
You are an AI ML debugging assistant.

System Data:
Profile: {profile}
Analysis: {analysis}
Issues: {issues}

User Question:
{question}

Answer clearly and technically.
"""

    response = call_llm(prompt)
    return response