from src.io import read_articles
import matplotlib.pyplot as plt
import networkx as net


articles = read_articles("./articles")

net.draw(articles[0].graph)
plt.show()

v = net.pagerank(articles[0].graph)

s = sorted(v.items(), key=lambda p: p[1], reverse=True)

print(s[0][0] + ' ' + s[1][0])
