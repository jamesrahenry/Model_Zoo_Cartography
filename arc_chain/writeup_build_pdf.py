"""Build the PDF of phase2/WRITEUP_POST_2026-10-28.md with figures embedded (pandoc + xelatex).  Written 2026-10-28.
usage: writeup_build_pdf.py [--out PDF]"""
import argparse, os, re, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.join(HERE, "..")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(P2, "figures", "writeup_2026-10-28", "ARC_Phase2_writeup_331937_trained_weights.pdf")); a = ap.parse_args()
src = open(os.path.join(P2, "WRITEUP_POST_2026-10-28.md"), encoding="utf-8").read()
figdir = os.path.abspath(os.path.join(P2, "figures", "writeup_2026-10-28"))
# title block
src = re.sub(r"^# \[Phase 2 write-up\] (.*)$", lambda m: "---\ntitle: >-\n  " + m.group(1) + "\nauthor: James Henry (jamesrahenry), with Claude Code\ndate: >-\n  ARC White-Box Estimation Challenge 2026, Phase 2 algorithmic-contribution write-up. 2026-10-28\n---\n", src, count=1, flags=re.M)
# drop the internal paste note
src = re.sub(r"^\*Written: 2026-10-28 .*?\*\n", "", src, count=1, flags=re.M | re.S)
# figure placeholders -> images (caption line that follows becomes the pandoc caption)
def fig(m):
    n, fn, cap = m.group(1), m.group(2), m.group(3).strip()
    cap = re.sub(r"^Figure \d+\.\s*", "", cap); w = "62%" if "alpha_closed_form" in fn else "100%"
    return f"![{cap}]({os.path.join(figdir, fn)}){{width={w}}}\n"
src = re.sub(r"^\[Figure (\d): `([^`]+)`\]\n\*(.*?)\*\n", fig, src, flags=re.M | re.S)
src = src.replace("`[LINK: estimator release]`", "[LINK: estimator release]")
src = src.replace("https://github.com/jamesrahenry/Model_Zoo_Cartography (MIT", "[the Model Zoo Cartography repository](https://github.com/jamesrahenry/Model_Zoo_Cartography) (MIT")
src = src.replace("| id | prediction | result |\n|---|---|---|", "| id | prediction | result |\n|:--------|:----------------------|:----------------------------------------------------|")
src = src.replace("| file | commit | sha256 (prefix) | written (UTC) |\n|---|---|---|---|", "| file | commit | sha256 (prefix) | written (UTC) |\n|:----------------------|:--------------|:------------------------|:----------------------|")
hdr = r"""
\PassOptionsToPackage{hyphens}{url}\usepackage{xurl}
\usepackage{float}\floatplacement{figure}{H}
\usepackage{caption}\captionsetup{font=small,labelfont=bf}
\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\emergencystretch}{3em}
"""
with tempfile.TemporaryDirectory() as td:
    md = os.path.join(td, "post.md"); open(md, "w", encoding="utf-8").write(src)
    h = os.path.join(td, "hdr.tex"); open(h, "w").write(hdr)
    cmd = ["pandoc", md, "-o", a.out, "--pdf-engine=xelatex", "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono",
           "-V", "fontsize=10pt", "-V", "geometry:margin=2cm", "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue", "--include-in-header", h, "--number-sections=false"]
    cmd = [c for c in cmd if c != "--number-sections=false"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit(1)
print("wrote", a.out)
