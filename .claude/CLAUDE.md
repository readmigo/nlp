# Readmigo NLP Project Guidelines

## Project Overview

Python NLP service for text processing and analysis.

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI application
│   ├── routers/         # API routes
│   ├── services/        # NLP services
│   └── models/          # Data models
├── scripts/             # Utility scripts
└── tests/               # Test cases
```

## Development Rules

### Tech Stack

- Language: Python 3.11+
- Framework: FastAPI
- NLP Libraries: spaCy, NLTK, jieba

### Development Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm

# Start development server
uvicorn app.main:app --reload
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /tokenize` | Tokenize text into words |
| `POST /sentences` | Split text into sentences |
| `POST /difficulty` | Assess word difficulty level |
| `POST /align` | Align bilingual paragraphs |

## Investigation & Problem Analysis

When investigating problems, output using this template:
```
问题的原因：xxx
解决的思路：xxx
修复的方案：xxx
```
