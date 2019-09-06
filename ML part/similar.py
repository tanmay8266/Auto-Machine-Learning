from nltk.corpus import wordnet
synonyms = []
antonyms = []

for syn in wordnet.synsets("tree"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))
