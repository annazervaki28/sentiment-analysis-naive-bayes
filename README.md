# Sentiment Analysis and Text Classification on Movie Reviews

## Description
This project focuses on building a Natural Language Processing (NLP) pipeline to classify movie reviews as either positive or negative (Sentiment Analysis). It explores various document representations, custom linguistic weighting techniques, and compares a custom Python implementation against the RapidMiner platform.

Developed for the Search Engines and Text Analysis course at the University of Macedonia.

## Repository Structure
* `anaforaPP2.pdf`: The comprehensive technical report (in Greek) detailing the methodology, preprocessing steps, linguistic rules, and performance comparisons.
* `pp2.py`: The Python source code containing the NLP preprocessing pipeline, custom weighting algorithms, and model evaluation.
* `pp2_rapidminer.rmp`: The RapidMiner process file used for the baseline comparison.

## Methodology & Feature Engineering
The project evaluates Naive Bayes classifiers (Multinomial, Bernoulli, Binarized Multinomial) across multiple text representations:
1. **Baseline Representations:** Evaluated Binary, Term Frequency (TF), TF-IDF, and Word2Vec embeddings (using mean pooling).
2. **Negation Handling:** Developed a custom function to append a `NOT_` prefix to tokens following negation words, capturing shifting sentiments.
3. **Part-of-Speech (POS) Weighting:** Utilized NLTK's `pos_tag` to double the weight of Adjectives and Adverbs, as they carry heavier sentiment.
4. **Positional Weighting:** Doubled the weight of tokens appearing in the first and last sentences of a review, assuming higher summary importance.

## Key Results
* **Best Model:** The **Binarized Multinomial Naive Bayes (Binary Representation)** achieved the highest baseline accuracy of **83.8%**.
* **Linguistic Impact:** Applying Part-of-Speech (POS) weighting proved to be the most effective custom linguistic technique, scoring 83.1%.
* **Python vs. RapidMiner:** The custom Python implementation significantly outperformed RapidMiner across all representations, beating the automated tool by **+16.55%** in Binary representation (83.8% vs 67.25%), highlighting the importance of tailored text preprocessing.

## Technologies Used
* Python
* NLTK (Tokenization, POS Tagging, Stopwords)
* Scikit-Learn (CountVectorizer, TfidfVectorizer, Naive Bayes models)
* RapidMiner
