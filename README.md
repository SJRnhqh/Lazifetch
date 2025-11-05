# Lazifetch

A Python package for searching and downloading academic literature based on topics.

> 📝 **Note**: This project is currently under active development.

## ✨ Features

- 🔍 Search academic papers by topic keywords
- 📥 Automatically download open access paper PDFs
- 📊 Smart ranking and filtering of search results
- 🤖 Semantic similarity-based reranking support

## 🚀 Quick Start

**Step 1**: Install Lazifetch

```shell
uv add lazifetch
```

**Step 2**: Install scipdf

```shell
uv pip install git+https://github.com/titipata/scipdf_parser

python -m spacy download en_core_web_sm
```

**Step 3**: Run grobid

Refer to the following process to install grobid, java should already be installed:

```shell
git clone https://github.com/kermitt2/grobid.git

cd grobid

./gradlew clean install

./gradlew run # should be run in background
```
