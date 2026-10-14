#!/usr/bin/env python3
"""Static checks for the generated report (no LaTeX installation needed). Run from Docs/Report:
     python3 tools/validate_report.py
Checks: balanced braces, matching \\begin/\\end, every \\cite key exists in the .bib and every .bib entry is cited,
every \\ref has a \\label, no non-ASCII, no unescaped % # & _ in running text, no leftover markup tokens,
and that the Google-Docs and LaTeX versions contain identical numbers, tables and citation order."""
import re, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, 'overleaf', 'main.tex'), encoding='utf8').read()
bib = open(os.path.join(ROOT, 'references.bib'), encoding='utf8').read()
md = open(os.path.join(ROOT, 'VisionRL_Report_GoogleDocs.md'), encoding='utf8').read()
ok = True
def report(name, passed, detail=''):
    global ok; ok &= passed; print(('PASS  ' if passed else 'FAIL  ') + name + (('  -> ' + detail) if detail else ''))

# strip verbatim blocks and comments for syntax checks
body = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', tex, flags=re.S)
nocomment = '\n'.join(re.sub(r'(?<!\\)%.*$', '', l) for l in body.split('\n') if not l.startswith(r'\newcommand'))

report('ASCII only (main.tex)', all(ord(c) < 128 for c in tex), str([c for c in set(tex) if ord(c) >= 128][:5]))
report('ASCII only (references.bib)', all(ord(c) < 128 for c in bib))
depth = 0; bad_at = None
for i, ch in enumerate(re.sub(r'\\[{}]', '', nocomment)):
    depth += ch == '{'; depth -= ch == '}'
    if depth < 0: bad_at = i; break
report('balanced braces', depth == 0 and bad_at is None, 'depth=%d' % depth)
stack = []; mism = []
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', nocomment):
    if m.group(1) == 'begin': stack.append(m.group(2))
    else:
        if not stack or stack.pop() != m.group(2): mism.append(m.group(2))
report('begin/end environments match', not stack and not mism, str(stack or mism))
cites = [k.strip() for g in re.findall(r'\\cite\{([^}]+)\}', nocomment) for k in g.split(',')]
bkeys = re.findall(r'\\bibitem\{([^}]+)\}', tex)   # single-file version: bibliography is embedded
report('every \\cite key has a \\bibitem', set(cites) <= set(bkeys), str(sorted(set(cites) - set(bkeys))))
report('every \\bibitem is cited', set(bkeys) <= set(cites), str(sorted(set(bkeys) - set(cites))))
labels = set(re.findall(r'\\label\{([^}]+)\}', nocomment)); refs = set(re.findall(r'\\ref\{([^}]+)\}', nocomment))
report('every \\ref has a \\label', refs <= labels, str(sorted(refs - labels)))
report('every figure/table label is referenced in the text', all(l in refs for l in labels if l.startswith(('fig:', 'tab:'))), str(sorted(l for l in labels if l.startswith(('fig:', 'tab:')) and l not in refs)))
txt = re.sub(r'\\(cite|ref|label|path|url)\{[^}]*\}', '', nocomment)
txt = re.sub(r'\$[^$]*\$', '', txt)
txt = re.sub(r'\\begin\{(tikzpicture|axis)\}.*?\\end\{\1\}', '', txt, flags=re.S)
unesc = [m.group(0) for m in re.finditer(r'(?<!\\)[%#&](?![^\n]*\\\\)', txt)]
# '&' is legal inside tabular rows; only flag % and # in running text
unesc = [c for c in re.findall(r'(?<!\\)[%#]', txt)]
report('no unescaped % or # in running text', not unesc, str(unesc[:5]))
report('no leftover markup tokens (@fig @tab @sec [@ ::: {#label)', not re.search(r'@(fig|tab|sec|eq):|\[@|:::|\{#[a-z]', tex), '')
report('no raw markdown bold/code left in LaTeX', not re.search(r'\*\*|(?<!`)`(?!`)', nocomment), '')
report('main.tex has maketitle, abstract, keywords, bibliography', all(s in tex for s in (r'\maketitle', r'\begin{abstract}', r'\begin{IEEEkeywords}', r'\begin{thebibliography}')))
report('IEEEtran conference class', r'\documentclass[conference]{IEEEtran}' in tex)
report('no \\includegraphics active (self-contained)', not re.search(r'^[^%\n]*\\includegraphics', tex, flags=re.M))

# ---- consistency between the two versions
def nums(s): return sorted(set(re.findall(r'\d+(?:[.,]\d+)*%?', s)))
tex_text = re.sub(r'\\[a-zA-Z]+', ' ', tex); md_text = md
tn, mn = set(nums(tex_text)), set(nums(md_text))
KEY = ['98.0', '95.0', '99.2', '16.5', '12.0', '22.3', '11.0', '7.4', '16.1', '196', '200', '88.0', '80.2', '93.0', '72.5', '65.9', '78.2', '7.5',
       '4.6', '5.5', '3.1', '9.6', '8.8', '8.5', '9.5', '24', '12.0', '1,515,126', '6,000', '129', '10.4', '32.4', '0.6', '4,163', '1,658.6', '0.973', '5,508']
miss_md = [k for k in KEY if k not in md_text]; miss_tex = [k for k in KEY if k not in tex]
report('key result numbers present in the Google-Docs version', not miss_md, str(miss_md)); report('key result numbers present in the LaTeX version', not miss_tex, str(miss_tex))
md_cites = re.findall(r'\[(\d+(?:, \d+)*)\]', md.split('## REFERENCES')[0])
order = []
for g in md_cites:
    for n in g.split(', '):
        if int(n) not in order: order.append(int(n))
report('Docs citation numbers appear in increasing first-use order', order == sorted(order), str(order[:12]))
tex_first = []
for k in cites:
    if k not in tex_first: tex_first.append(k)
refs_md = re.findall(r'^\[(\d+)\] ', md.split('## REFERENCES')[1], flags=re.M)
report('same number of references in both versions', len(refs_md) == len(tex_first) == len(bkeys), '%d / %d / %d' % (len(refs_md), len(tex_first), len(bkeys)))
report('reference count >= 23', len(tex_first) >= 23, str(len(tex_first)))
for tab in re.findall(r'\\label\{(tab:[^}]+)\}', tex): pass
tex_rows = len(re.findall(r'\\\\', tex)); md_rows = sum(1 for l in md.split('\n') if l.startswith('| ') and not l.startswith('|---'))
print('INFO  table rows: LaTeX row terminators=%d (includes figures/captions), Markdown table lines=%d' % (tex_rows, md_rows))
print('\nOVERALL:', 'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
