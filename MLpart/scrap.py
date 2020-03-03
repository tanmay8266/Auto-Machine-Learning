import requests
from bs4 import BeautifulSoup

query = "blood%20%circulation%20%in%20%heart"
url = 'https://scholar.google.com/scholar?q=' + query + '&ie=UTF-8&oe=UTF-8&hl=en&btnG=Search'

content = requests.get(url).text
page = BeautifulSoup(content, 'lxml')
results = []
for entry in page.find_all("h3", attrs={"class": "gs_rt"}):
    results.append({"title": entry.a.text, "url": entry.a['href']})
print(results)