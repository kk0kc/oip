import os
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def build_inverted_index_txt(pages_dir='pages', output_file='inverted_index.txt'):
    inverted_index = defaultdict(list)

    for page_dir in os.listdir(pages_dir):
        if not page_dir.startswith('page_'):
            continue

        page_num = int(page_dir.split('_')[1])
        lemmas_path = os.path.join(pages_dir, page_dir, 'lemmas.txt')

        if os.path.exists(lemmas_path):
            with open(lemmas_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    lemma = parts[0]
                    inverted_index[lemma].append(page_num)

    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted(inverted_index.keys()):
            doc_ids = ','.join(map(str, sorted(set(inverted_index[term]))))
            f.write(f"{term}:{doc_ids}\n")

    return inverted_index


inverted_index = build_inverted_index_txt()