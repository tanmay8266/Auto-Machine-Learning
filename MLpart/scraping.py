from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

stop_words.add('the')

pitch = input()
word_tokens = word_tokenize(pitch) 
  
filtered = [w for w in word_tokens if not w in stop_words] 
  
filtered = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered.append(w) 
  
print(word_tokens) 
print(filtered)

string = ""
for word in filtered:
    string = string + word + "%20%"
'''string = list(string)
print(string)
string.pop()
string = ''.join(string)'''
print(string)
query = string
'''synonyms = []

for syn in wordnet.synsets("tall"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        

synonyms = set(synonyms)

print(synonyms)

for word in synonyms:
    query = word
    print(query)'''
url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q='+query+'&btnG='

print(url)
content = requests.get(url).text
page = BeautifulSoup(content, 'lxml')
results = []
for entry in page.find_all("h3", attrs={"class": "gs_rt"}):
    results.append({"title": entry.a.text, "url": entry.a['href']})
compressed = results[0:5]
print(compressed)