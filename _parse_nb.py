import json

p = r"c:\Users\rauni\OneDrive\Documents\Learning\Projects\ads-intelligence-platform\notebooks\01_data_understanding.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", [])).strip().replace("\n", " ")[:120]
    ctype = c["cell_type"]
    outs = len(c.get("outputs", []))
    print(f"{i:2d} {ctype:8s} outs={outs:2d} | {src}")
