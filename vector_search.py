import os
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class VectorSearch:
    def __init__(self, pages_dir='pages', index_path='inverted_index.txt'):
        self.pages_dir = pages_dir
        self.lemmas_map = self._build_lemmas_map()
        self.inverted_index = self._load_inverted_index(index_path)
        self.doc_vectors, self.doc_norms = self._load_tfidf_vectors()
        self.common_lemmas = self._identify_common_lemmas()

    def _build_lemmas_map(self) -> Dict[str, str]:
        """Строит словарь {слово: лемма} из всех файлов lemmas.txt"""
        lemmas_map = {}
        for page_dir in os.listdir(self.pages_dir):
            if not page_dir.startswith('page_'):
                continue

            lemmas_path = os.path.join(self.pages_dir, page_dir, 'lemmas.txt')
            if not os.path.exists(lemmas_path):
                continue

            with open(lemmas_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    lemma = parts[0]
                    for word in parts[1:]:
                        lemmas_map[word] = lemma
        return lemmas_map

    def _load_inverted_index(self, index_path='inverted_index.txt') -> Dict[str, List[int]]:
        """Загружает обратный индекс из файла"""
        index = defaultdict(list)
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                lemma, doc_ids_str = line.strip().split(":")
                doc_ids = list(map(int, doc_ids_str.split(",")))
                index[lemma] = doc_ids
        return dict(index)

    def _identify_common_lemmas(self) -> Set[str]:
        """Определяет леммы, встречающиеся во всех документах"""
        total_docs = len([d for d in os.listdir(self.pages_dir) if d.startswith('page_')])
        return {lemma for lemma, docs in self.inverted_index.items()
                if len(docs) == total_docs}

    def _load_tfidf_vectors(self) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
        """Загружает TF-IDF векторы и вычисляет их нормы"""
        doc_vectors = {}
        doc_norms = {}

        for page_dir in os.listdir(self.pages_dir):
            if not page_dir.startswith('page_'):
                continue

            doc_id = int(page_dir.split('_')[1])
            tfidf_path = os.path.join(self.pages_dir, page_dir, 'lemmas_tfidf.txt')
            doc_vectors[doc_id] = {}

            if not os.path.exists(tfidf_path):
                continue

            with open(tfidf_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    lemma, _, tfidf = parts[0], parts[1], float(parts[2])
                    doc_vectors[doc_id][lemma] = tfidf

            # Вычисляем норму вектора документа
            doc_norms[doc_id] = math.sqrt(sum(
                tfidf ** 2 for tfidf in doc_vectors[doc_id].values()
            )) if doc_vectors[doc_id] else 0.0

        return doc_vectors, doc_norms

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """Поиск с учетом общих терминов"""
        words = re.findall(r'\w+', query.lower())
        query_lemmas = [self.lemmas_map[word]
                        for word in words
                        if word in self.lemmas_map]

        if not query_lemmas:
            print("Не найдено лемм для поиска.")
            return []


        # Строим вектор запроса
        query_vector = {}
        for lemma in set(query_lemmas):
            if lemma in self.common_lemmas:
                # Для общих лемм используем TF (частоту в запросе)
                query_vector[lemma] = query_lemmas.count(lemma) / len(query_lemmas)
            else:
                # Для остальных лемм используем бинарный вес
                query_vector[lemma] = 1

        query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))

        # Находим релевантные документы
        relevant_docs = set()
        for lemma in query_lemmas:
            if lemma in self.inverted_index:
                relevant_docs.update(self.inverted_index[lemma])

        # Вычисляем косинусную близость
        results = []
        for doc_id in relevant_docs:
            if doc_id not in self.doc_vectors:
                continue

            dot_product = 0.0
            for lemma, q_weight in query_vector.items():
                if lemma in self.doc_vectors[doc_id]:
                    doc_weight = self.doc_vectors[doc_id][lemma]
                    dot_product += q_weight * doc_weight

            doc_norm = self.doc_norms[doc_id]
            if doc_norm == 0 or query_norm == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = min(dot_product / (query_norm * doc_norm), 1.0)

            if cosine_sim > 1e-6:
                results.append((doc_id, cosine_sim))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


    def interactive_search(self):
        """Интерактивный режим поиска"""
        print("=== Интеллектуальная поисковая система ===")
        print(f"Загружено документов: {len(self.doc_vectors)}")
        print(f"Общих лемм: {len(self.common_lemmas)}")
        print("Введите 'exit' для выхода\n")

        while True:
            query = input("Поисковый запрос: ").strip()
            if query.lower() == 'exit':
                break

            results = self.search(query)

            if not results:
                print("Ничего не найдено.")
                continue

            print("\nТоп результатов:")
            for doc_id, score in results:
                print(f"Документ {doc_id}: релевантность = {score:.4f}")
                print()


if __name__ == '__main__':
    search_engine = VectorSearch()
    search_engine.interactive_search()
