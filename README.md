# Intelligent Hotel Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-semantic%20search-00A98F)](https://www.sbert.net/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-cosine%20similarity-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000)](https://ollama.com/)

An AI-powered hotel search and recommendation application developed for the **Introduction to Artificial Intelligence** course of the **MSc in Artificial Intelligence at the University of Minho (UMinho)**.

## Problem and solution

Searching hotel reviews by manually matching free-text requirements is slow and does not capture semantic intent. This project turns a natural-language request such as *“I want an affordable hotel in London near Big Ben for a business trip”* into a structured search and a ranked recommendation.

The solution combines:

- multilingual sentence embeddings for semantic matching between the request and hotel reviews;
- local LLM-based extraction of city, amenities, points of interest, price intent, and trip type;
- deterministic fallback extraction when the LLM is unavailable;
- metadata and review filters for city, amenities, trip type, and price preference;
- a grounded LLM explanation based only on the shortlisted hotels.

## Features and technical architecture

### Main features

- Natural-language hotel queries in Portuguese or English.
- Entity and intent extraction for:
  - destination city;
  - amenities such as pool, Wi-Fi, breakfast, and gym;
  - points of interest such as Big Ben, Eiffel Tower, metro, or city centre;
  - business, family, couple, solo, and leisure trips;
  - budget or luxury preferences.
- Configurable amenity filters and 1–5 recommendation results.
- Semantic ranking with cosine similarity over precomputed review embeddings.
- Strict filtering when matching metadata exists, with score boosts/penalties for softer preferences.
- Duplicate removal at hotel level, preserving the highest-scoring review.
- Interactive PyDeck map using hotel latitude and longitude.
- Local LLM-generated recommendation and short hotel descriptions through Ollama.

### Processing pipeline

```text
Hotel_Reviews.csv
        |
        v
pre_processamento_v2.ipynb
  cleaning, sampling, NLP/NER,
  city and POI extraction,
  sentence embeddings
        |
        v
Hotel_Reviews_processed.pkl / .csv
        |
        v
app.py (Streamlit)
  query understanding -> filtering -> semantic ranking
  -> map visualisation -> grounded recommendation
```

The preprocessing notebook cleans the original reviews, keeps a balanced sample per hotel, normalises tags, extracts keywords and named entities with spaCy, identifies the supported cities, and generates embeddings with `paraphrase-multilingual-MiniLM-L12-v2`. The application reuses those embeddings and calculates query-to-review cosine similarity with scikit-learn.

## Installation and execution

### Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and available on `PATH`
- The `mistral` Ollama model
- The original Hotel Reviews dataset

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Diogodias28/AI-intro.git
cd AI-intro

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn sentence-transformers ollama pydeck spacy nltk jupyter
python -m spacy download en_core_web_sm
```

### 3. Prepare the dataset and processed artefacts

Download `Hotel_Reviews.csv` from the [dataset source on Kaggle](https://www.kaggle.com/code/ahmedshahriarsakib/text-classification-classic-ml-nn-tf-hub/input) and place it in the repository root.

Run all cells in [`pre_processamento_v2.ipynb`](./pre_processamento_v2.ipynb). The notebook downloads the NLTK English stopword corpus when needed and generates:

- `Hotel_Reviews_processed.pkl` — preferred by the application because it preserves the embedding arrays;
- `Hotel_Reviews_processed.csv` — portable CSV alternative.

The generated data files are intentionally ignored by Git because of their size.

### 4. Configure and start Ollama

Start the Ollama service, then download the model used by `app.py`:

```bash
ollama pull mistral
```

Keep Ollama running while using the application. The app calls the local model for query interpretation and the final recommendation; semantic retrieval and the fallback parser still provide the core search behaviour if an LLM response cannot be parsed.

### 5. Run the Streamlit application

From the repository root:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`. Enter a request in the search box, optionally select amenity filters, and choose the number of recommendations.

## Project structure

```text
.
├── app.py                       # Streamlit UI and recommendation pipeline
├── pre_processamento_v2.ipynb   # Current dataset preparation pipeline
├── pre_processamento.ipynb      # Earlier preprocessing notebook
├── .gitignore                   # Excludes local datasets and generated artefacts
└── README.md
```

## Authorship

This repository contains a collaborative academic project developed by students in the **MSc in Artificial Intelligence at the University of Minho (UMinho)** for the **Introduction to Artificial Intelligence** course. The implementation combines the group's work on data preparation, NLP, semantic retrieval, local LLM integration, and the Streamlit interface.
