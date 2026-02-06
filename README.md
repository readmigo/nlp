# Readmigo NLP

[![CI](https://github.com/readmigo/nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/readmigo/nlp/actions/workflows/ci.yml)

Natural Language Processing service for Readmigo.

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **NLP Libraries**: spaCy, NLTK, jieba

## Features

- Text tokenization and analysis
- Sentence boundary detection
- Word difficulty assessment
- Chinese text segmentation
- Bilingual text alignment

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

## Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
