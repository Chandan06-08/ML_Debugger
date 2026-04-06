from groq import Groq
import os
from dotenv import load_dotenv

# 1. INITIALIZE
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. LLM CLIENT
def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# 3. PROMPT BUILDER
def build_prompt(profile, analysis, issues):
    return f"""
You are an expert ML engineer.

Dataset Profile:
{profile}

Training Analysis:
{analysis}

Detected Issues:
{issues}

Give output STRICTLY in this format:

Issue:
<short issue>

Reason:
- point 1
- point 2

Fix:
- fix 1
- fix 2

Keep it clean and structured.
"""

# 4. RAG KNOWLEDGE BASE
KNOWLEDGE_BASE = {
    "overfitting": "Overfitting occurs when the model learns noise in the training data, leading to a high training accuracy (e.g., > 90%) but a much lower validation accuracy (e.g., < 70%). Common fixes include: 1. Reducing model complexity. 2. Adding Dropout layers (0.2 - 0.5). 3. Early stopping. 4. Increasing training data size.",
    "underfitting": "Underfitting is when the model is too simple to capture the underlying pattern. Train and Val accuracy are both low. Fixes: 1. Use a more complex model (e.g., DecisionTree instead of Logistic). 2. Increase the number of epochs. 3. Add more features (feature engineering).",
    "class imbalance": "Data balance is critical for reliable evaluation. If one class has significantly more samples (e.g., > 2:1 ratio), SMOTE or Oversampling should be used. Evaluation metrics like F1-score or AUC-PR are more informative than Accuracy in this case.",
    "low accuracy": "General low performance can be due to noisy data, lack of features, or sub-optimal hyperparameters. Scaling (StandardScaler) or better feature selection is often required."
}

def retrieve_context(issues):
    context = []
    for issue in issues:
        if issue in KNOWLEDGE_BASE:
            context.append(f"- {issue.upper()}: {KNOWLEDGE_BASE[issue]}")
    return "\n\n".join(context) if context else "No specialized knowledge retrieved."

# 5. AUTO FIX ENGINE
def generate_fixes(issues):
    recommendations = []
    if "overfitting" in issues:
        recommendations.append("Reduce model complexity (e.g., lower Decision Tree depth)")
        recommendations.append("Add more training data if possible")
        recommendations.append("Introduce L1/L2 regularization")
    if "underfitting" in issues:
        recommendations.append("Use a more complex algorithm (e.g., SVM or Gradient Boosting)")
        recommendations.append("Train for more epochs or relax regularization")
    if "class imbalance" in issues:
        recommendations.append("Use Oversampling (SMOTE) or Undersampling to balance dataset")
        recommendations.append("Switch from Accuracy to F1-score or AUC-PR")
    if "low accuracy" in issues:
        recommendations.append("Standardize/Scale numerical features (StandardScaler)")
    return recommendations if recommendations else ["No specific automated fixes available."]

# 6. CONVERSATIONAL CHAT
def ask_question(question, profile=None, analysis=None, issues=None):
    rag_context = retrieve_context(issues or [])
    prompt = f"""
You are an expert AI ML Debugging Assistant.
Your goal is to explain technical issues and suggest fixes based on the data provided.

--- SYSTEM CONTEXT ---
DATA PROFILE: {profile if profile else "No profile data available."}
ANALYSIS: {analysis if analysis else "No training analysis available."}
DETECTED ISSUES: {issues if issues else "No specific issues detected."}

--- RAG KNOWLEDGE BASE ---
{rag_context}

--- USER QUESTION ---
{question}

--- INSTRUCTIONS ---
- Use the RAG knowledge and system context to provide a technically accurate answer.
- If the question is about fixing an issue, be specific.
- Keep the tone professional but clear.
"""
    return call_llm(prompt)
