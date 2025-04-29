from flask import Flask, render_template, request
from vector_search import VectorSearch

app = Flask(__name__)

pages_dir = 'pages'
index_path = 'inverted_index.txt'
search_engine = VectorSearch(pages_dir=pages_dir, index_path=index_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""

    if request.method == 'POST':
        query = request.form['query'].strip()
        if query:
            search_results = search_engine.search(query, top_n=10)
            results = [
                {
                    "doc_id": doc_id,
                    "score": f"{score:.4f}",
                }
                for doc_id, score in search_results
            ]

    return render_template('index.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)