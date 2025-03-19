import os
import random
import time
import requests
from bs4 import BeautifulSoup

if not os.path.exists('pages'):
    os.makedirs('pages')


base_url = 'https://www.frontiersin.org'
search_url = 'https://www.frontiersin.org/journals/public-health/articles'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]
article_links = []

def get_article_links(url, max_pages=10):
    for page in range(1, max_pages + 1):
        page_url = f'{url}?page={page}'
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if '/articles/' in link['href'] and 'doi.org' not in link['href']:
                full_link = base_url + link['href'] if not link['href'].startswith('http') else link['href']
                if full_link not in article_links:
                    article_links.append(full_link)
        print(f'Processed page {page}, found {len(article_links)} articles so far.')
        time.sleep(random.uniform(1, 3))
    return article_links

def download_page(url, index):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open(f'pages/page_{index}.html', 'w', encoding='utf-8') as file:
            file.write(response.text)
    except requests.exceptions.RequestException as e:
        print(f'Failed to download {url}: {e}')

get_article_links(search_url)

for index, link in enumerate(article_links[:150]):
    download_page(link, index)
    with open('index.txt', 'a') as index_file:
        index_file.write(f'{index}: {link}\n')
    print(f'Downloaded {index}: {link}')