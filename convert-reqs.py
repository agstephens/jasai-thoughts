#!/usr/bin/env python
# turn YAML into a requirements.txt, then pip install
import sys, yaml
y = yaml.safe_load(open("requirements.yml"))

outfile = "requirements.txt"
with open(outfile, "w") as writer:
    writer.write("\n".join(sorted(y["packages"]) + ["\n"]))

print(f"[INFO] Wrote {outfile}")

