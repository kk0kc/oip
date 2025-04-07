import re
import operator
from collections import defaultdict
from nltk.stem import WordNetLemmatizer


class BooleanSearchEngine:
    def __init__(self, index_file='inverted_index.txt'):
        self.index = defaultdict(list)
        self.lemmatizer = WordNetLemmatizer()
        self.operations = {
            'AND': operator.and_,
            'OR': operator.or_,
            'NOT': operator.sub
        }
        self._load_index(index_file)

    def _load_index(self, index_file):
        """
        Загрузка инвертированного индекса из текстового файла

        Формат файла:
        term:doc_id1,doc_id2,doc_id3
        """
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                term, doc_ids = line.split(':', 1)
                self.index[term] = list(map(int, doc_ids.split(',')))

    def _shunting_yard(self, tokens):
        output = []
        operators = []
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}

        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            elif token in precedence:
                while (operators and operators[-1] != '(' and
                       precedence[operators[-1]] >= precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            else:
                output.append(token.lower())

        while operators:
            output.append(operators.pop())

        return output

    def _evaluate_postfix(self, postfix):
        """
        Вычисление постфиксного выражения

        :param postfix: список токенов в постфиксной нотации
        :return: список ID документов, удовлетворяющих запросу
        """
        stack = []

        for token in postfix:
            if token in self.operations:
                if token == 'NOT':
                    operand = set(stack.pop())
                    all_docs = set(range(self._get_total_documents()))
                    result = all_docs - operand
                else:
                    operand2 = set(stack.pop())
                    operand1 = set(stack.pop())
                    result = self.operations[token](operand1, operand2)
                stack.append(list(result))
            else:
                lemma = self.lemmatizer.lemmatize(token)
                stack.append(self.index.get(lemma, []))

        return sorted(stack[0]) if stack else []

    def _get_total_documents(self):
        max_doc_id = 0
        for doc_ids in self.index.values():
            if doc_ids:
                max_doc_id = max(max_doc_id, max(doc_ids))
        return max_doc_id + 1

    def search(self, query):
        """
        Выполнение поиска по булевому запросу

        :param query: строка запроса (например, "(cat AND dog) OR (mouse NOT cheese)")
        :return: отсортированный список ID документов
        """
        if not query.strip():
            return []

        tokens = re.findall(r'\(|\)|[\w]+', query)
        postfix = self._shunting_yard(tokens)
        return self._evaluate_postfix(postfix)

    def interactive_search(self):
        print("Boolean Search Engine")
        print("Supported operators: AND, OR, NOT")
        print("Parentheses can be used for grouping")
        print("Example: (cat AND dog) OR (mouse NOT cheese)")
        print("Type 'exit' to quit\n")

        while True:
            query = input("Search query: ").strip()
            if query.lower() == 'exit':
                break

            if not query:
                continue

            try:
                results = self.search(query)
                if results:
                    print(f"Found {len(results)} documents:")
                    print(", ".join(map(str, results)))
                else:
                    print("No documents found")
                print()
            except Exception as e:
                print(f"Error processing query: {e}\n")


if __name__ == '__main__':
    search_engine = BooleanSearchEngine('inverted_index.txt')
    search_engine.interactive_search()