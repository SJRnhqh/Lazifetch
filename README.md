# Lazifetch

[![PyPI version](https://badge.fury.io/py/lazifetch.svg)](https://badge.fury.io/py/lazifetch)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI downloads](https://img.shields.io/pypi/dm/lazifetch.svg)](https://pypi.org/project/lazifetch/)

A Python package for searching and downloading academic literature based on topics — designed to streamline literature review and idea discovery.

> 📝 **Note**: This project is currently under active development. Breaking changes may occur.

## 🌟 Overview

Lazifetch helps researchers quickly find, rank, and download relevant academic papers using keyword queries. It integrates semantic search, metadata filtering, and PDF parsing to deliver a seamless "fetch-and-read" experience — ideal for early-stage exploration or integration into LLM-based research agents (e.g., inspired by frameworks like [Chain-of-Ideas](https://github.com/DAMO-NLP-SG/CoI-Agent)).

## ✨ Features

- 🔍 **Search academic papers** by topic keywords (via Semantic Scholar API)
- 📥 **Automatically download** open-access PDFs
- 📊 **Smart ranking** with configurable filters (year, citation count, venue)
- 🤖 **Optional semantic similarity reranking** using embeddings
- 📄 **Extract structured content** (title, abstract, references) from PDFs via GROBID
- ⚙️ **Lightweight and scriptable** — perfect for automation or agent workflows
- 🚀 **Async support** for efficient concurrent operations

## 🚀 Quick Start

### Step 1: Install Lazifetch

We recommend using [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
uv add lazifetch
```

Or with pip:

```bash
pip install lazifetch
```

### Step 2: Install PDF parsing dependencies

Lazifetch uses [scipdf_parser](https://github.com/titipata/scipdf_parser) + [GROBID](https://github.com/kermitt2/grobid) for high-quality PDF extraction:

```bash
uv pip install git+https://github.com/titipata/scipdf_parser
python -m spacy download en_core_web_sm
```

### Step 3: Set up GROBID service

GROBID must be running as a service for PDF parsing. You have two options:

#### Option A: Run GROBID locally

```bash
git clone https://github.com/kermitt2/grobid.git
cd grobid
./gradlew clean install
./gradlew run  # Keep this running in the background (default: http://localhost:8070)
```

#### Option B: Use Dockerized GROBID

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Step 4: Configure API key

Create a `.env` file in your project root and add your Semantic Scholar API key:

```bash
SEMANTIC_SEARCH_API_KEY=your_api_key_here
```

You can get a free API key from [Semantic Scholar API](https://www.semanticscholar.org/product/api).

### Step 5: Use Lazifetch in your code

```python
import asyncio
from lazifetch import SemanticSearcher

async def main():
    # Initialize the searcher
    searcher = SemanticSearcher(
        save_dir="./papers",  # Directory to save downloaded PDFs
        ban_list=[]  # Optional: list of paper titles to exclude
    )
    
    # Search and download papers
    results = await searcher.search_async(
        query="large language models for scientific discovery",
        max_results=5,
        year=2020,  # Optional: filter by year
        need_download=True,  # Set to False to only get metadata
        api_key="your_api_key"  # Or load from environment variable
    )
    
    # Process results
    for paper in results:
        print(f"📄 {paper.title} ({paper.year})")
        print(f"   Citations: {paper.citations_count}")
        print(f"   Abstract: {paper.abstract[:100]}...")
        print()

asyncio.run(main())
```

## 📚 Advanced Usage

### Semantic Reranking

To use semantic similarity reranking, you need to provide an LLM object with an `get_embedding()` method:

```python
class YourLLM:
    def get_embedding(self, texts: list[str]) -> list[list[float]]:
        # Return embeddings for each text
        pass

llm = YourLLM()

results = await searcher.search_async(
    query="machine learning",
    max_results=10,
    rerank_query="deep learning for computer vision",  # Different query for reranking
    llm=llm,
    api_key="your_api_key"
)
```

### Search with Filters

```python
results = await searcher.search_async(
    query="neural networks",
    max_results=20,
    year=2023,  # Filter by publication year
    minCitationCount=10,  # Minimum citation count
    publicationDate="2023-01-01",  # Publication date filter
    fieldsOfStudy=["Computer Science"],  # Filter by field of study
    api_key="your_api_key"
)
```

### Read Paper Content

After downloading papers, you can extract structured content:

```python
# Get full paper content with references
for paper in results:
    if paper.article:
        content = searcher.read_paper_content_with_ref(paper.article)
        print(content)
```

### Search Related Papers

Find papers that cite or are cited by a specific paper:

```python
related_papers = await searcher.search_related_paper_async(
    title="Your Paper Title",
    need_citation=True,  # Include papers that cite this paper
    need_reference=True,  # Include papers cited by this paper
    api_key="your_api_key"
)
```

## 📖 API Reference

### `SemanticSearcher`

Main class for searching and downloading papers.

#### Parameters

- `save_dir` (str): Directory path to save downloaded PDFs. Default: `"papers/"`
- `ban_list` (List[str]): List of paper titles to exclude from search results. Default: `[]`

#### Methods

##### `search_async(query, max_results=5, ...)`

Search for papers and optionally download them.

**Parameters:**

- `query` (str): Search query string
- `max_results` (int): Maximum number of results to return
- `year` (int, optional): Filter by publication year
- `publicationDate` (str, optional): Filter by publication date
- `need_download` (bool): Whether to download PDFs. Default: `True`
- `rerank_query` (str, optional): Query for semantic reranking
- `llm` (Any, optional): LLM object with `get_embedding()` method for reranking
- `api_key` (str, optional): Semantic Scholar API key

**Returns:** List of `Result` objects

##### `search_related_paper_async(title, ...)`

Search for papers related to a specific paper.

**Parameters:**

- `title` (str): Title of the paper to find related papers for
- `need_citation` (bool): Include papers that cite this paper
- `need_reference` (bool): Include papers cited by this paper
- `rerank_query` (str, optional): Query for semantic reranking
- `llm` (Any, optional): LLM object for reranking
- `api_key` (str, optional): Semantic Scholar API key

**Returns:** List of `Result` objects

### `Result`

Result object representing a paper.

#### Attributes

- `title` (str): Paper title
- `abstract` (str): Paper abstract
- `article` (dict, optional): Parsed article content from PDF
- `citations_count` (int): Number of citations
- `year` (int, optional): Publication year

## 🧠 Inspiration & Acknowledgements

This project was partly inspired by recent advances in LLM-powered research agents, particularly:

- Li, Long et al. "Chain of Ideas: Revolutionizing Research via Novel Idea Development with LLM Agents", arXiv:2410.13185 (2024).
  - 🔗 [Paper](https://arxiv.org/abs/2410.13185) | [Official Code](https://github.com/DAMO-NLP-SG/CoI-Agent)

We thank the authors for pioneering the vision of autonomous idea generation and literature synthesis.

Additionally, we rely on:

- [Semantic Scholar API](https://www.semanticscholar.org/product/api) for paper search
- [GROBID](https://github.com/kermitt2/grobid) for PDF structuring
- [scipdf_parser](https://github.com/titipata/scipdf_parser) for Python integration

## ⚠️ Limitations

- Only downloads open-access papers (due to copyright restrictions)
- GROBID must be running locally for full-text parsing
- Search coverage depends on backend APIs (Semantic Scholar)
- Rate limits may apply to Semantic Scholar API (free tier: 100 requests per 5 minutes)

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

> **Note**: Ensure compliance with the terms of use of underlying services (e.g., Semantic Scholar API, GROBID).

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

## 📝 Requirements

- Python >= 3.12
- GROBID service running (local or Docker)
- Semantic Scholar API key (optional but recommended)
