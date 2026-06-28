# SmartPayroll AI

> End-to-end AI and Data Science system for HR analytics and payroll intelligence.
> Built on Microsoft Azure AI Foundry — from raw data ingestion to production AI agents.

---

## Architecture

```
Raw Data (IBM HR Analytics CSV)
         │
         ▼ Bronze → Silver → Gold (ETL Pipeline)
         │
         ├──▶ Classical ML (XGBoost + MLflow)
         │         └── Attrition prediction: AUC 0.772
         │
         ├──▶ RAG Pipeline (Azure AI Foundry + Phi-4-mini-instruct)
         │         └── HR policy Q&A with semantic search
         │
         ├──▶ AI Investigation Agent (4 tools)
         │         └── Employee risk analysis and reporting
         │
         └──▶ FastAPI Service
                   └── REST API: 4 endpoints, Swagger UI
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud | Microsoft Azure AI Foundry (Sweden Central) |
| Chat Model | Phi-4-mini-instruct (Microsoft) |
| Embedding Model | text-embedding-3-small (OpenAI via Azure) |
| Data Engineering | Pandas, Pandera, PyArrow, Parquet |
| Architecture | Medallion (Bronze → Silver → Gold) |
| Machine Learning | Scikit-learn, XGBoost, SMOTE, MLflow |
| RAG | Cosine similarity retrieval, semantic chunking (500 tokens) |
| Agents | Custom tool-calling investigation agent |
| API | FastAPI, Pydantic v2, Uvicorn |
| CI/CD | GitHub Actions, feature branches, pull requests |
| Version Control | Git — conventional commits, no direct push to main |

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 employees × 35 features
- Target: Attrition (16.1% positive — imbalanced classification)
- Licence: Public domain

---

## Key Results

| Component | Metric | Result |
|-----------|--------|--------|
| Logistic Regression (baseline) | AUC-ROC | 0.772 |
| Logistic Regression (baseline) | F1 Score | 0.411 |
| XGBoost + SMOTE | AUC-ROC | 0.750 |
| XGBoost + SMOTE | F1 Score | 0.490 |
| RAG retrieval | Top chunk similarity | 0.72+ |
| Agent risk scan | HIGH risk found (top 50) | 5 employees |

---

## EDA Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | Class Imbalance | 16.1% attrition — requires SMOTE or class_weight |
| 2 | Overtime = #1 Risk | Overtime workers 2.9x more likely to leave (30.5% vs 10.4%) |
| 3 | Salary Gap | Leavers earn $2,002/month less than stayers |
| 4 | Experience Protects | TotalWorkingYears strongest negative correlation (-0.171) |
| 5 | Job Level Matters | Higher job level = lower attrition probability (-0.169) |
| 6 | Distance Effect | Distance from home positively correlated with leaving (+0.078) |

---

## Project Structure

```
smartpayroll-ai/
│
├── src/
│   ├── data/
│   │   ├── ingest.py              # Load CSV from local/Azure Blob Storage
│   │   ├── validate.py            # Pandera schema validation (5 checks)
│   │   ├── clean.py               # Bronze → Silver transformation
│   │   └── pipeline.py            # Pipeline orchestrator
│   │
│   ├── features/
│   │   └── feature_engineering.py # Encoding, scaling, train/test split
│   │
│   ├── models/
│   │   └── attrition/
│   │       └── train.py           # LR baseline + XGBoost + MLflow tracking
│   │
│   ├── rag/
│   │   ├── document_processor.py  # Chunk + embed HR policy documents
│   │   └── chain.py               # Cosine similarity retrieval + generation
│   │
│   ├── agents/
│   │   ├── tools/
│   │   │   └── hr_tools.py        # 3 tools: details, risk, dept stats
│   │   └── investigation_agent.py # Full investigation + batch risk scan
│   │
│   └── api/
│       └── main.py                # FastAPI service with 4 endpoints
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA with 8 cells and 4 charts
│
├── hr_policies/
│   ├── HR_Policy_Annual_Leave_ES.md  # Spanish annual leave policy
│   └── HR_Policy_Overtime.md         # General overtime policy
│
├── docs/
│   ├── eda_01_attrition.png       # Attrition distribution chart
│   ├── eda_02_overtime.png        # Overtime vs attrition chart
│   ├── eda_03_salary.png          # Salary distribution chart
│   └── eda_04_correlation.png     # Feature correlation heatmap
│
├── data/
│   ├── raw/                       # Original CSV (gitignored)
│   └── processed/                 # Silver layer parquet (gitignored)
│
├── .env.example                   # Environment variable template
├── .gitignore                     # Protects secrets and data files
├── requirements.txt               # All dependencies pinned
├── pyproject.toml                 # ruff + black + pytest config
└── README.md                      # This file
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Najam0786/smartpayroll-ai.git
cd smartpayroll-ai

# 2. Create virtual environment (Python 3.11 required)
py -3.11 -m venv smartpayroll
smartpayroll\Scripts\activate        # Windows
# source smartpayroll/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env and add your Azure credentials

# 5. Download dataset
# From: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Save as: data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

# 6. Run the data pipeline
python -m src.data.pipeline

# 7. Train the ML model
python -m src.models.attrition.train

# 8. Process HR policy documents (requires Azure credentials)
python -m src.rag.document_processor

# 9. Run the investigation agent
python -m src.agents.investigation_agent

# 10. Start the FastAPI service
python -m src.api.main
# Open: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/employees/{id}` | Get employee details |
| POST | `/api/v1/attrition/risk` | Get attrition risk assessment |
| POST | `/api/v1/investigate` | Run full investigation report |

### Example: Attrition Risk

```bash
curl -X POST http://localhost:8000/api/v1/attrition/risk \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 7}'
```

Response:
```json
{
  "employee_id": 7,
  "risk_level": "HIGH",
  "risk_score": 9,
  "risk_factors": [
    "Working overtime — 2.9x higher attrition risk",
    "Low job satisfaction: 1/4",
    "Short tenure: 1 years",
    "Below median salary: $2,670"
  ],
  "recommendation": "Immediate retention conversation recommended",
  "department": "Research & Development",
  "monthly_income": 2670.0
}
```

---

## Engineering Standards

- **Branch strategy**: Feature branches only — no direct push to main
- **Pull requests**: Every change reviewed before merge
- **Commit style**: Conventional commits (feat/fix/chore/docs/refactor)
- **Security**: Azure credentials in .env only — never committed
- **Data**: Raw and processed data gitignored — never in version control
- **Python**: 3.11.9 in isolated virtual environment
- **Dependencies**: All packages pinned in requirements.txt

---

## Azure Setup

```
Project:    smartpayroll-ai (Microsoft Azure AI Foundry)
Region:     Sweden Central
Models deployed:
  - Phi-4-mini-instruct    (chat completion)
  - text-embedding-3-small (embeddings, 1536 dimensions)
```

---

## Author

**Nazmul Farooquee**
AI & Data Science Engineer
Barcelona, Spain

[![GitHub](https://img.shields.io/badge/GitHub-Najam0786-black?logo=github)](https://github.com/Najam0786)
