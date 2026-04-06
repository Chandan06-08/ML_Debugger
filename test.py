from app.core.profiler import profile_data
from app.core.analyzer import analyze_training
from app.core.detector import detect_issues
from app.llm.prompts import build_prompt
from app.llm.llm_client import call_llm

profile = profile_data("data/sample_dataset.csv")
analysis = analyze_training("data/real_logs.json")
issues = detect_issues(profile, analysis)

prompt = build_prompt(profile, analysis, issues)
explanation = call_llm(prompt)

print("PROFILE:", profile)
print("\nANALYSIS:", analysis)
print("\nISSUES:", issues)
print("\nEXPLANATION:\n", explanation)