import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng') 



def get_baseline_tokens(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]

def get_negation_text(text):
    sentences = sent_tokenize(text.lower())
    neg_words = {'not', 'no', 'never', "don't", "didn't", "can't", "couldn't"}
    all_processed = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        neg_mode = False
        for t in tokens:
            if t in neg_words:
                neg_mode = True
                all_processed.append(t)
            elif t in ".,!?;":
                neg_mode = False
            else:
                all_processed.append(f"NOT_{t}" if neg_mode else t)
    
    return " ".join([t for t in all_processed if t.replace('NOT_', '').isalpha()])

# sinartiseis stathmisis b meros

def apply_pos_weights(matrix, feature_names):
    """ Β.β: Διπλασιασμός βάρους για Επίθετα και Επιρρήματα """
    new_matrix = matrix.toarray().astype(float)
    tags = pos_tag(feature_names)
    for i, (word, tag) in enumerate(tags):
        if tag.startswith('JJ') or tag.startswith('RB'):
            new_matrix[:, i] *= 2
    return new_matrix

def apply_positional_weights(texts, vectorizer):
    """ Β.γ: Διπλασιασμός βάρους για λέξεις στην 1η και τελευταία πρόταση """
    matrix = vectorizer.transform(texts).toarray().astype(float)
    feature_map = vectorizer.vocabulary_
    
    for row_idx, text in enumerate(texts):
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            # proti kai teleftaia protasi
            target_sents = [sentences[0], sentences[-1]]
            for sent in target_sents:
                words = get_baseline_tokens(sent)
                for w in words:
                    if w in feature_map:
                        col_idx = feature_map[w]
                        matrix[row_idx, col_idx] *= 2
    return matrix

def load_data():
    raw_texts, labels = [], []
    base_path = 'txt_sentoken'
    for label, category in enumerate(['neg', 'pos']):
        dir_path = os.path.join(base_path, category)
        for fname in os.listdir(dir_path):
            with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                raw_texts.append(f.read())
                labels.append(label)
    return raw_texts, np.array(labels)

def evaluate(clf, X, y, desc):
    scoring = {'acc': 'accuracy', 'f1': make_scorer(f1_score, average='macro')}
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
    print(f"{desc:45} | Acc: {np.mean(scores['test_acc']):.3f} | F1: {np.mean(scores['test_f1']):.3f}")

# main

if __name__ == "__main__":
    print("Φόρτωση δεδομένων...")
    raw_data, labels = load_data()
    
    # pro epeksergamsena keimena gia to baseline
    baseline_texts = [" ".join(get_baseline_tokens(t)) for t in raw_data]

    print("\n--- ΜΕΡΟΣ Α: BASELINE ---")
    v_bin = CountVectorizer(binary=True)
    X_bin = v_bin.fit_transform(baseline_texts)
    evaluate(BernoulliNB(), X_bin, labels, "K2+E1: Bernoulli NB (Binary)")
    evaluate(MultinomialNB(), X_bin, labels, "K3+E1: Binarized Multinomial NB")

    v_tf = CountVectorizer()
    X_tf = v_tf.fit_transform(baseline_texts)
    evaluate(MultinomialNB(), X_tf, labels, "K1+E2: Multinomial NB (TF)")

    v_tfidf = TfidfVectorizer()
    X_tfidf = v_tfidf.fit_transform(baseline_texts)
    evaluate(MultinomialNB(), X_tfidf, labels, "K1+E3: Multinomial NB (TF-IDF)")

    print("\n--- ΜΕΡΟΣ Β: ΓΛΩΣΣΙΚΕΣ ΤΕΧΝΙΚΕΣ ---")
    
    #Ba: Negation
    neg_texts = [get_negation_text(t) for t in raw_data]
    X_neg = CountVectorizer().fit_transform(neg_texts)
    evaluate(MultinomialNB(), X_neg, labels, "B.a: Baseline + Negation")

    # Bb POS weights
    # periorizw features se 2000 
    v_pos = CountVectorizer(max_features=2000)
    X_pos_raw = v_pos.fit_transform(baseline_texts)
    X_pos_weighted = apply_pos_weights(X_pos_raw, v_pos.get_feature_names_out())
    evaluate(MultinomialNB(), X_pos_weighted, labels, "B.b: Baseline + POS Weights (x2)")

    # Βg Sentence Position Weights
    v_posi = CountVectorizer(max_features=2000)
    v_posi.fit(baseline_texts)
    X_posi_weighted = apply_positional_weights(raw_data, v_posi)
    evaluate(MultinomialNB(), X_posi_weighted, labels, "B.g: Baseline + Sentence Position (x2)")

    print("\n--- ΜΕΡΟΣ Β.5: ΣΥΝΔΥΑΣΜΟΣ ΟΛΩΝ ---")
    # sindiazw Negation (Strings) me POS Weights
    v_all = CountVectorizer(max_features=2000)
    X_all_raw = v_all.fit_transform(neg_texts) 
    X_all_weighted = apply_pos_weights(X_all_raw, v_all.get_feature_names_out())
    evaluate(MultinomialNB(), X_all_weighted, labels, "B.5: All Techniques Combined")

    from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

print("\n--- ΑΝΑΛΥΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΓΙΑ ΤΟΝ ΚΑΛΥΤΕΡΟ ΚΑΤΗΓΟΡΙΟΠΟΙΗΤΗ (K3+E1) ---")

# xwrismos dedomenwn se 80% ekpaidefsi kai 20% dokimi ( confusion matrix)
X_train, X_test, y_train, y_test = train_test_split(X_bin, labels, test_size=0.2, random_state=42)

#ekpaidefsi modelou
best_clf = MultinomialNB()
best_clf.fit(X_train, y_train)

# provlepseis
y_pred = best_clf.predict(X_test)

#  confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

#  rrecision, recall, F1 (Macroaverage)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))