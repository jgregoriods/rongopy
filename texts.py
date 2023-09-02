import os


# Read all the texts
texts = []
for filename in os.listdir("rapa_nui_texts"):
    with open(f"rapa_nui_texts/{filename}", "r") as f:
        lines = f.readlines()
        for l in lines:
            if "[" not in l and l[:-1]:
                texts.append(l[:-1])

texts = list(set(texts))
texts.sort()
