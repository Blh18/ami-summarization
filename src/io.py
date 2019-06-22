import os
import nltk
from model.document import Document
from src.graph import create_graph, process_document, count_tfidf

# load all articles from a folder and process them
# also computes tfidfs and creates graph of sentences
def read_articles(path):
    articles = []
    cnt = 0

    for filename in os.listdir(path):
        with open(path + '/' + filename) as file:
            doc = Document(cnt, file.read())
            doc.words, doc.sentences = process_document(doc.raw_text)
            count_tfidf(doc.sentences, doc.words)
            doc.graph = create_graph(doc.sentences, doc.words)
            cnt = cnt + 1
            articles.append(doc)

    return articles
