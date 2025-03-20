import os
import re
import spacy
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import words

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

english_words = set(words.words())
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def tokenize(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens

def lemmatize(tokens):
    lemmas = {}
    for token in tokens:
        doc = nlp(token)
        lemma = doc[0].lemma_
        if lemma not in lemmas:
            lemmas[lemma] = set()
        lemmas[lemma].add(token)
    return lemmas

def process_documents(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            page_name = filename.replace('.html', '')
            output_dir = os.path.join(directory, page_name)
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text()
                cleaned_text = clean_text(text)
                tokens = tokenize(cleaned_text)
                filtered_tokens = set()
                for token in tokens:
                    if (token in english_words and
                        token not in stop_words and
                        len(token) > 2 and
                        nlp(token)[0].pos_ != 'PROPN'):
                        filtered_tokens.add(token)
                lemmas = lemmatize(filtered_tokens)

                with open(os.path.join(output_dir, 'tokens.txt'), 'w', encoding='utf-8') as f:
                    for token in sorted(filtered_tokens):
                        f.write(f"{token}\n")
                with open(os.path.join(output_dir, 'lemmas.txt'), 'w', encoding='utf-8') as f:
                    for lemma, words in sorted(lemmas.items()):
                        if lemma in english_words:
                            f.write(f"{lemma} {' '.join(sorted(words))}\n")

            print(f"Processed {filename} and saved results to {output_dir}")
process_documents('pages')