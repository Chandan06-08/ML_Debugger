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