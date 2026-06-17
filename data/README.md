# Data

This directory is intentionally empty of source content.

## Getting the data

This project uses the **Merck Manual Professional Edition** as its knowledge base.
The manual is freely available online at no cost — no account or PDF required.

Run the scraper to fetch the relevant sections:

```bash
conda activate anaconda-clinical-rag
python scripts/scraper.py
```

This will populate `data/raw/` with clean text files for 7 clinical topics
across 5 domains (Critical Care, Gastrointestinal, Neurology, Dermatology,
Orthopedics), respecting the site's 5-second crawl delay.

Then build the FAISS index:

```bash
python scripts/build_index.py
```

This populates `data/index/` with the vector store and chunk metadata.

## What gets scraped

| Domain | Topics |
|---|---|
| Critical Care | Sepsis & Septic Shock, Shock, ARDS |
| Gastrointestinal | Appendicitis |
| Neurology | Traumatic Brain Injury |
| Dermatology | Alopecia Areata |
| Orthopedics | Compartment Syndrome |

## Legal note

Content is sourced from [merckmanuals.com/professional](https://www.merckmanuals.com/professional),
which is publicly accessible and permits crawling per its robots.txt (crawl-delay: 5).
This project is for educational and demonstration purposes only.
The scraped data files are not committed to this repository.
