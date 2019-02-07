from src.io import read_articles

articles = read_articles("./articles")

print(articles[0].raw_text)
