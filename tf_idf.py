import os
import math
from collections import defaultdict


def calculate_tf_idf(pages_dir='pages'):
    doc_terms = defaultdict(dict)
    doc_lemmas = defaultdict(dict)
    doc_lengths = defaultdict(int)
    term_docs = defaultdict(set)
    lemma_docs = defaultdict(set)

    for page_dir in os.listdir(pages_dir):
        if not page_dir.startswith('page_'):
            continue

        doc_id = int(page_dir.split('_')[1])
        doc_path = os.path.join(pages_dir, page_dir)

        # Обработка терминов из tokens.txt
        tokens_path = os.path.join(doc_path, 'tokens.txt')
        if os.path.exists(tokens_path):
            with open(tokens_path, 'r', encoding='utf-8') as f:
                terms = [line.strip() for line in f if line.strip()]
                for term in terms:
                    doc_terms[doc_id][term] = doc_terms[doc_id].get(term, 0) + 1
                    term_docs[term].add(doc_id)
                doc_lengths[doc_id] += len(terms)

        # Обработка лемм из lemmas.txt
        lemmas_path = os.path.join(doc_path, 'lemmas.txt')
        if os.path.exists(lemmas_path):
            with open(lemmas_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    lemma = parts[0]
                    count = len(parts) - 1
                    doc_lemmas[doc_id][lemma] = doc_lemmas[doc_id].get(lemma, 0) + count
                    lemma_docs[lemma].add(doc_id)

    # 2. Расчет IDF
    total_docs = len(doc_lengths)

    def safe_idf(df, N):
        return math.log((N) / (df))

    # IDF для терминов
    term_idf = {term: safe_idf(len(docs), total_docs)
                for term, docs in term_docs.items()}

    # IDF для лемм
    lemma_idf = {lemma: safe_idf(len(docs), total_docs)
                 for lemma, docs in lemma_docs.items()}

    # 3. Расчет и сохранение TF-IDF
    for doc_id in doc_lengths:
        doc_dir = os.path.join(pages_dir, f'page_{doc_id}')

        # Для терминов (из tokens.txt)
        with open(os.path.join(doc_dir, 'terms_tfidf.txt'), 'w', encoding='utf-8') as f:
            for term, count in doc_terms[doc_id].items():
                tf = count / doc_lengths[doc_id]
                tf_idf = tf * term_idf[term]
                f.write(f"{term} {term_idf[term]:.6f} {tf_idf:.6f}\n")

        # Для лемм (из lemmas.txt)
        with open(os.path.join(doc_dir, 'lemmas_tfidf.txt'), 'w', encoding='utf-8') as f:
            for lemma, count in doc_lemmas[doc_id].items():
                tf = count / doc_lengths[doc_id]
                tf_idf = tf * lemma_idf[lemma]
                f.write(f"{lemma} {lemma_idf[lemma]:.6f} {tf_idf:.6f}\n")


if __name__ == '__main__':
    calculate_tf_idf()