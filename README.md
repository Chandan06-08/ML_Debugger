# 🚀 AI-Powered ML System Debugger

An intelligent, rule-based, and LLM-driven diagnostic tool for Machine Learning pipelines. This system goes beyond simple error messages—it **profiles** your data, **analyzes** your training stability, **detects** systemic failures, and uses **RAG (Retrieval-Augmented Generation)** to provide technical explanations and actionable fixes.

---

## ✨ Key Features

- **📊 Smart Data Profiling**: Automatically extracts dataset statistics, missing value reports, and class distribution checks.
- **📈 Training Diagnostics**: Real-time analysis of training vs. validation accuracy to identify overfitting or underfitting.
- **🧠 AI Explainability (Groq-Powered)**: Integrates with **Llama 3.1** via Groq to provide expert-level technical reasoning.
- **📚 Basic RAG System**: Uses a built-in knowledge base of ML best practices to ground the AI's "thinking" in verified technical facts.
- **🛠️ Auto-Fix Recommendation**: Suggests specific code and architecture changes (like SMOTE, L2 Regularization, or depth limits) for every detected issue.
- **💬 Natural Language Interface**: A dedicated chat interface that understands the context of your specific model run.

---

## 🏗️ Project Architecture

The project follows a "Simplified Core" architecture, designed to be clean and easy for developers to extend:

```text
ML Debugger/
├── app/core/           # [Protected] The ML Engine logic
│   ├── analyzer.py     # Training log stability analyzer
│   ├── detector.py     # Rule-based issue detection
│   └── profiler.py     # Data statistics extraction
├── main.py             # FastAPI Server & API Endpoints
├── ai_logic.py         # Consolidated LLM, RAG, & Fixer logic
├── index.html          # Clean, responsive Web UI
├── Data/               # (Git Ignored) Training logs and CSVs
├── .env                # (Git Ignored) API Credentials
└── requirements.txt    # Project dependencies
```

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Analysis**: Pandas, Scikit-Learn, NumPy
- **Intelligence**: Groq SDK (Llama-3.1-8b-instant)
- **Frontend**: Vanilla HTML5, CSS3 (Modern UI), JavaScript

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed and a **Groq API Key**. You can get one at [console.groq.com](https://console.groq.com/).

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Chandan06-08/ML_Debugger.git
cd ML_Debugger

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your key:
```env
GROQ_API_KEY=your_key_here
```

### 4. Run the Debugger
```powershell
python main.py
```
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

---

## 📖 How it Works (RAG System)

The debugger doesn't just guess. When an issue (like "overfitting") is detected by the **Core Engine**, the **RAG System** retrieves specific technical documentation from the `KNOWLEDGE_BASE` in `ai_logic.py`. This documentation is then injected into the LLM's prompt, ensuring the explanation you get is accurate, grounded, and free of hallucinations.

---

## 📝 License
Proprietary / MIT (Check repository for details)

---

> [!TIP]
> Use the **"Ask Questions"** box at the bottom of the UI to ask follow-up questions like *"Why is my gap so high?"* or *"How do I implement the suggested fix?"*—the AI remembers your model's specific data!
