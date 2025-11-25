import json

data = []

with open("recipe_corpus_full.json", "r") as infile:
    for line in infile:
        data.append(json.loads(line))

samples = data[:10000]

with open("recipe_corpus_sample.json", "w", encoding="utf-8") as outfile:
    json.dump(samples, outfile, ensure_ascii=False, indent=4)