#!/usr/bin/env python3
"""Builds both report deliverables from ONE source so that text, numbers and citation order are identical.

  tools/paper_source.md + references.bib  ->  VisionRL_Report_GoogleDocs.md   (paste into Google Docs)
                                          ->  overleaf/main.tex + overleaf/references.bib  (compile in Overleaf)

Run:  python3 tools/build_report.py      (from Docs/Report, standard library only)
"""
import re, sys, os, shutil
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(HERE, 'paper_source.md'), encoding='utf8').read().replace('\\_', '_')
BIB = open(os.path.join(ROOT, 'references.bib'), encoding='utf8').read()

# ------------------------------------------------------------------ bibliography
def parse_bib(text):
    entries = OrderedDict()
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, flags=re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', body):
            fields[fm.group(1).lower()] = fm.group(2)
        entries[key] = (typ, fields)
    return entries
BIBDB = parse_bib(BIB)

def strip_tex(s):
    s = re.sub(r"\{\\[`'^\"~=.]([a-zA-Z])\}", r'\1', s)       # {\`a} -> a
    s = s.replace('--', '–').replace('{', '').replace('}', '')
    return s

def fmt_authors(a):
    names = [x.strip() for x in re.split(r'\s+and\s+', a)]
    out, etal = [], False
    for n in names:
        if n == 'others': etal = True; continue
        n = strip_tex(n)
        if ',' in n:
            last, first = [p.strip() for p in n.split(',', 1)]
            initials = ' '.join(p[0] + '.' for p in re.split(r'[\s-]+', first) if p)
            out.append(f'{initials} {last}')
        else:
            out.append(n)
    if etal: return ', '.join(out) + ', et al.'
    if len(out) == 1: return out[0]
    if len(out) == 2: return f'{out[0]} and {out[1]}'
    return ', '.join(out[:-1]) + ', and ' + out[-1]

def fmt_ref_docs(key):
    typ, f = BIBDB[key]
    au = fmt_authors(f['author']); t = strip_tex(f['title'])
    yr = f.get('year', '')
    if typ == 'book':
        return f"{au}, *{t}*, {f.get('edition','')} ed. {f.get('address','')}: {f.get('publisher','')}, {yr}.".replace(' ed.  ', ' ed. ')
    if typ == 'article':
        s = f"{au}, \"{t},\" *{strip_tex(f['journal'])}*, vol. {f['volume']}, no. {f['number']}, pp. {strip_tex(f['pages'])}, {yr}"
        if 'doi' in f: s += f", doi: {f['doi']}"
        return s + '.'
    if typ == 'inproceedings':
        s = f"{au}, \"{t},\" in *{strip_tex(f['booktitle'])}*, {yr}"
        if 'pages' in f: s += f", pp. {strip_tex(f['pages'])}"
        if 'note' in f: s += f", {f['note']}"
        return s + '.'
    hp = f.get('howpublished', '')
    if hp.startswith('arXiv'): return f"{au}, \"{t},\" {hp}, {yr}."
    return f"{au}, \"{t}.\" {hp}"

# ------------------------------------------------------------------ block parser
def parse_blocks(src):
    lines = src.split('\n'); i = 0; blocks = []; para = []
    def flush():
        nonlocal para
        if para: blocks.append(('para', ' '.join(para))); para = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('::: '):
            flush(); head = ln[4:]; body = []; i += 1
            while lines[i].strip() != ':::': body.append(lines[i]); i += 1
            blocks.append(('block', head, body))
        elif ln.startswith('#'):
            flush(); m = re.match(r'(#+\*?)\s+(.*?)(?:\s*\{#([\w:]+)\})?\s*$', ln)
            blocks.append(('head', m.group(1), m.group(2), m.group(3)))
        elif re.match(r'^- ', ln):
            flush(); items = []
            while i < len(lines) and re.match(r'^- ', lines[i]): items.append(lines[i][2:].strip()); i += 1
            blocks.append(('ul', items)); continue
        elif re.match(r'^\d+\. ', ln):
            flush(); items = []
            while i < len(lines) and re.match(r'^\d+\. ', lines[i]): items.append(re.sub(r'^\d+\.\s+', '', lines[i]).strip()); i += 1
            blocks.append(('ol', items)); continue
        elif ln.strip() == '': flush()
        else: para.append(ln.strip())
        i += 1
    flush(); return blocks

def parse_attrs(text):
    d = {}
    for m in re.finditer(r'(\w+)=("([^"]*)"|\S+)|(?:^|\s)(wide)(?=\s|$)', text):
        if m.group(4): d['wide'] = True
        else: d[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(2)
    return d

BLOCKS = parse_blocks(SRC)

# ------------------------------------------------------------------ numbering (same order in both outputs)
def cites_in(text): return [k.strip().lstrip('@') for grp in re.findall(r'\[@([^\]]+)\]', text) for k in grp.split(';')]
CITE_ORDER = []
for k in cites_in(SRC):
    if k not in CITE_ORDER: CITE_ORDER.append(k)
missing = [k for k in CITE_ORDER if k not in BIBDB]
if missing: sys.exit('CITED BUT NOT IN BIB: %s' % missing)
unused = [k for k in BIBDB if k not in CITE_ORDER]
if unused: sys.exit('IN BIB BUT NEVER CITED: %s' % unused)
CITENUM = {k: n + 1 for n, k in enumerate(CITE_ORDER)}

LABELS = {}; sec_counter = [0, 0, 0]; fig_n = tab_n = eq_n = 0
def roman(n): return ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'][n]
for b in BLOCKS:
    if b[0] == 'head':
        lvl = len(b[1].rstrip('*'))
        if b[1].endswith('*'): continue
        if lvl == 1: sec_counter[0] += 1; sec_counter[1] = 0; num = roman(sec_counter[0])
        elif lvl == 2: sec_counter[1] += 1; num = roman(sec_counter[0]) + '-' + chr(64 + sec_counter[1])
        else: continue
        if b[3]: LABELS[b[3]] = num
    elif b[0] == 'block':
        name = b[1].split()[0]; attrs = parse_attrs(b[1] + ' ' + ' '.join(l for l in b[2] if not re.match(r'^[DHR]:', l)))
        if name == 'figure': fig_n += 1; LABELS[attrs['id']] = str(fig_n)
        elif name == 'table': tab_n += 1; LABELS[attrs['id']] = roman(tab_n)
        elif name == 'equation': eq_n += 1; LABELS[attrs['id']] = str(eq_n)

# ------------------------------------------------------------------ inline converters
UNI_TEX = {'×': r'$\times$', '−': r'$-$', '–': '--', '—': '---', '≥': r'$\geq$', '≤': r'$\leq$', '≈': r'$\approx$',
           '→': r'$\rightarrow$', 'γ': r'$\gamma$', 'Φ': r'$\Phi$', '±': r'$\pm$', 'λ': r'$\lambda$', 'β': r'$\beta$',
           'ε': r'$\epsilon$', '’': "'", '‘': "'", '“': '``', '”': "''", '·': r'$\cdot$', '…': r'\ldots{}'}
TEX_DOCS = [(r'\times', '×'), (r'\geq', '≥'), (r'\leq', '≤'), (r'\approx', '≈'), (r'\gamma', 'γ'), (r'\Phi', 'Φ'), (r'\lambda', 'λ'),
            (r'\epsilon', 'ε'), (r'\beta', 'β'), (r'\pm', '±'), (r'\rightarrow', '→'), (r'\cdot', '·'), (r'\ldots', '…'), (r'\,', ' '),
            (r'\mathbb{1}', '1'), (r'\qquad', '   ')]

def esc_tex(s):
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('~', r'\textasciitilde{}'), ('^', r'\^{}')): s = s.replace(a, b)
    return s

def to_tex(s, allow_path=True):
    holds = []
    def hold(x): holds.append(x); return '\x00%d\x00' % (len(holds) - 1)
    s = re.sub(r'\$([^$]+)\$', lambda m: hold('$' + m.group(1) + '$'), s)
    def code(m):
        c = m.group(1)
        if allow_path and ('/' in c or ('.' in c and ' ' not in c)): return hold(r'\url{' + c + '}')
        return hold(r'\texttt{' + esc_tex(c) + '}')
    s = re.sub(r'`([^`]+)`', code, s)
    s = re.sub(r'\[@([^\]]+)\]', lambda m: hold(r'\cite{' + ','.join(k.strip().lstrip('@') for k in m.group(1).split(';')) + '}'), s)
    s = re.sub(r'Section @(sec:[\w]+)', lambda m: hold(r'Section~\ref{' + m.group(1) + '}'), s)
    s = re.sub(r'@(fig:[\w]+)', lambda m: hold(r'Fig.~\ref{' + m.group(1) + '}'), s)
    s = re.sub(r'@(tab:[\w]+)', lambda m: hold(r'Table~\ref{' + m.group(1) + '}'), s)
    s = re.sub(r'@(eq:[\w]+)', lambda m: hold(r'(\ref{' + m.group(1) + '})'), s)
    s = re.sub(r'"([^"]+)"', lambda m: '“' + m.group(1) + '”', s)
    s = esc_tex(s)
    for u, t in UNI_TEX.items(): s = s.replace(u, t)
    s = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])', r'\\textit{\1}', s)
    return re.sub(r'\x00(\d+)\x00', lambda m: holds[int(m.group(1))], s)

def math_to_docs(m):
    t = m
    t = re.sub(r'\\text\{([^}]*)\}', r'\1', t)
    for a, b in TEX_DOCS: t = t.replace(a, b)
    t = re.sub(r'_\{([^}]*)\}', r'_\1', t)
    t = t.replace('{', '').replace('}', '').replace('\\', '')
    return t

def to_docs(s):
    s = re.sub(r'\$([^$]+)\$', lambda m: math_to_docs(m.group(1)), s)
    s = re.sub(r'\[@([^\]]+)\]', lambda m: '[' + ', '.join(str(CITENUM[k.strip().lstrip('@')]) for k in m.group(1).split(';')) + ']', s)
    s = re.sub(r'@sec:([\w]+)', lambda m: LABELS['sec:' + m.group(1)], s)
    s = re.sub(r'@fig:([\w]+)', lambda m: 'Fig. ' + LABELS['fig:' + m.group(1)], s)
    s = re.sub(r'@tab:([\w]+)', lambda m: 'Table ' + LABELS['tab:' + m.group(1)], s)
    s = re.sub(r'@eq:([\w]+)', lambda m: '(' + LABELS['eq:' + m.group(1)] + ')', s)
    return s

# ------------------------------------------------------------------ TikZ / pgfplots figures (LaTeX only)
TIKZ = {}
TIKZ['arch'] = r'''\resizebox{0.97\textwidth}{!}{%
\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\scriptsize, minimum height=1.0cm, minimum width=3.2cm, fill=blue!7},
  gbox/.style={box, fill=green!10},
  arr/.style={->, >=Stealth, thick},
  lab/.style={font=\tiny, align=center}]
\node[box] (env) at (0,0) {Unity scene\\(room, robot, exit)};
\node[box] (cam) at (4.4,0) {Camera sensor\\84$\times$84 RGB frame};
\node[box] (pol) at (8.8,0) {Policy network\\(CNN encoder + MLP)};
\node[box] (act) at (13.2,0) {Action\\(4 discrete moves)};
\draw[arr] (env) -- (cam);
\draw[arr] (cam) -- node[above,lab]{image} (pol);
\draw[arr] (pol) -- node[above,lab]{sample} (act);
\draw[arr] (act.north) -- ++(0,0.7) -| node[pos=0.25,above,lab]{one-cell move} (env.north);
\node[gbox] (rew) at (0,-2.6) {Reward and\\termination};
\node[gbox] (trn) at (8.8,-2.6) {PPO trainer\\(Python, PyTorch)};
\node[gbox] (onx) at (13.2,-2.6) {ONNX export $\rightarrow$\\Unity inference engine};
\draw[arr] (env) -- node[right,lab]{state} (rew);
\draw[arr] (rew) -- node[above,lab]{reward, done} (trn);
\draw[arr] (trn) -- node[right,lab]{weight updates} (pol);
\draw[arr] (trn) -- node[above,lab]{export} (onx);
\end{tikzpicture}}'''
TIKZ['workflow'] = r'''\begin{tikzpicture}[
  box/.style={draw, rounded corners=2pt, align=center, font=\scriptsize, minimum height=0.85cm, minimum width=6.4cm, fill=blue!7},
  gate/.style={box, fill=orange!15},
  arr/.style={->, >=Stealth, thick}]
\node[box]  (a) at (0,0)    {1. Environment tests\\(mechanics, rewards, resets)};
\node[gate] (b) at (0,-1.35) {2. Fidelity gate: $\geq$99\% frame--state agreement\\(if it fails: diagnose, do not train)};
\node[box]  (c) at (0,-2.7) {3. PPO training (200k steps, 21 training exits)};
\node[box]  (d) at (0,-3.9) {4. ONNX export};
\node[box]  (e) at (0,-5.1) {5. Python evaluation\\(camera / blank camera, baselines)};
\node[box]  (f) at (0,-6.45) {6. Unity runtime validation\\(Burst CPU, sampled actions)};
\foreach \p/\q in {a/b,b/c,c/d,d/e,e/f} \draw[arr] (\p) -- (\q);
\end{tikzpicture}'''

def pgf_bars():
    rows = [('Uniform', 5.5, 3.1, 9.6), ('Blind', 16.5, 12.0, 22.3), ('C50k', 72.5, 65.9, 78.2), ('B50k', 7.5, 4.6, 12.0), ('Fcam', 98.0, 95.0, 99.2), ('Fblank', 11.0, 7.4, 16.1)]
    coords = ' '.join('(%s,%.1f) += (0,%.1f) -= (0,%.1f)' % (n, v, hi - v, v - lo) for n, v, lo, hi in rows)
    return (r'''\begin{tikzpicture}
\begin{axis}[ybar, width=\columnwidth, height=5.4cm, bar width=8pt, ymin=0, ymax=105, ylabel={Success rate (\%)},
  symbolic x coords={Uniform,Blind,C50k,B50k,Fcam,Fblank}, xtick=data,
  xticklabels={Uniform,Blind,50k cam.,50k blank,Final cam.,Final blank},
  xticklabel style={rotate=35, anchor=east, font=\tiny}, yticklabel style={font=\tiny}, ylabel style={font=\scriptsize},
  enlarge x limits=0.12, ymajorgrids, grid style={dotted}]
\addplot+[fill=blue!35, draw=blue!60!black, error bars/.cd, y dir=both, y explicit] coordinates {''' + coords + r'''};
\end{axis}
\end{tikzpicture}''')

RUN02 = [(10, -0.937), (20, -0.679), (30, 0.034), (40, 0.244), (50, 0.393), (60, 0.852), (70, 1.174), (80, 1.288), (90, 1.328), (100, 1.339),
         (110, 1.349), (120, 1.354), (130, 1.351), (140, 1.360), (150, 1.346), (160, 1.356), (170, 1.355), (180, 1.358), (190, 1.352), (200, 1.350)]
RUN01 = [(5, -0.873), (10, -0.819), (15, -0.888), (20, -0.884), (25, -0.814), (30, -0.931), (35, -0.800), (40, -0.749), (45, -0.904), (50, -0.728),
         (55, -0.781), (60, -0.976), (65, -0.704), (70, -0.871), (75, -0.816), (80, -0.877), (85, -0.841), (90, -0.922), (95, -0.795), (100, -0.792), (105, -0.839)]
def pgf_curve():
    c2 = ' '.join('(%d,%.3f)' % p for p in RUN02); c1 = ' '.join('(%d,%.3f)' % p for p in RUN01)
    return (r'''\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=5.0cm, xlabel={Training steps ($\times 10^{3}$)}, ylabel={Mean episode reward},
  xmin=0, xmax=200, ymin=-1.2, ymax=1.6, grid=both, grid style={dotted}, tick label style={font=\tiny}, label style={font=\scriptsize},
  legend style={font=\tiny, at={(0.98,0.45)}, anchor=east}]
\addplot[blue, thick, mark=*, mark size=1pt] coordinates {''' + c2 + r'''};
\addlegendentry{Run 02 (corrected player)}
\addplot[red, thick, mark=square*, mark size=1pt] coordinates {''' + c1 + r'''};
\addlegendentry{Run 01 (stale frames)}
\end{axis}
\end{tikzpicture}''')

# ------------------------------------------------------------------ LaTeX emission
def tex_table(attrs, rows):
    align = re.sub(r'p\{([^}]*)\}', r'>{\\raggedright\\arraybackslash}p{\1}', attrs.get('align', 'll'))
    env = 'table*' if attrs.get('wide') else 'table'
    out = [r'\begin{%s}[!t]' % env, r'\caption{%s}' % to_tex(attrs['caption'], allow_path=False), r'\label{%s}' % attrs['id'],
           r'\centering\footnotesize\renewcommand{\arraystretch}{1.12}', r'\begin{tabular}{@{}%s@{}}' % align, r'\toprule']
    hdr = [r for r in rows if r[0] == 'H'][0][1]
    out.append(' & '.join(to_tex(c.strip(), False) for c in hdr.split('|')) + r' \\ \midrule')
    for kind, line in rows:
        if kind == 'R': out.append(' & '.join(to_tex(c.strip(), False) for c in line.split('|')) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{%s}' % env]
    return '\n'.join(out)

def tex_figure(attrs):
    wide = attrs.get('wide'); env = 'figure*' if wide else 'figure'
    kind = attrs['kind']; out = [r'\begin{%s}[!t]' % env, r'\centering']
    if kind.startswith('tikz:'): out.append(TIKZ[kind[5:]])
    elif kind == 'pgf:bars': out.append(pgf_bars())
    elif kind == 'pgf:curve': out.append(pgf_curve())
    else:
        width = r'0.95\textwidth' if wide else r'0.92\columnwidth'; height = '3.4cm' if wide else '3.0cm'
        out.append(r'%% To use the real image, upload it to a figures/ folder and replace the next line with:')
        out.append(r'%% \includegraphics[width=%s]{figures/%s}' % (width.replace('0.95', '0.95').replace('0.92', '0.92'), os.path.basename(attrs['file'])))
        out.append(r'\figplaceholder{%s}{%s}{\textbf{PLACEHOLDER} -- insert: \texttt{%s}}' % (height, width, esc_tex(os.path.basename(attrs['file']))))
    out += [r'\caption{%s}' % to_tex(attrs['caption'], allow_path=False), r'\label{%s}' % attrs['id'], r'\end{%s}' % env]
    return '\n'.join(out)

PREAMBLE = r'''\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% ---- packages -------------------------------------------------------------
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
% Placeholder box for screenshots that are not supplied (keeps the project compilable without image files).
\newcommand{\figplaceholder}[3]{\fbox{\parbox[c][#1][c]{#2}{\centering\footnotesize #3}}}

\begin{document}

\title{@@TITLE@@}

\author{\IEEEauthorblockN{[Student Name 1]}
\IEEEauthorblockA{Department of [Department]\\
[Institution]\\
[City, Country]\\
[email1@example.edu]}
\and
\IEEEauthorblockN{[Student Name 2]}
\IEEEauthorblockA{Department of [Department]\\
[Institution]\\
[City, Country]\\
[email2@example.edu]}
\and
\IEEEauthorblockN{[Student Name 3]}
\IEEEauthorblockA{Department of [Department]\\
[Institution]\\
[City, Country]\\
[email3@example.edu]}
% Add or delete \and blocks to match the number of authors.
}

\maketitle

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{IEEEkeywords}
@@KEYWORDS@@
\end{IEEEkeywords}

'''
POSTAMBLE = r'''

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
'''

def build_tex():
    title = abstract = keywords = ''; body = []
    for b in BLOCKS:
        if b[0] == 'head':
            lvl = len(b[1].rstrip('*')); star = '*' if b[1].endswith('*') else ''
            cmd = {1: 'section', 2: 'subsection', 3: 'subsubsection'}[lvl]
            line = r'\%s%s{%s}' % (cmd, star, to_tex(b[2], False)) + (r'\label{%s}' % b[3] if b[3] else '')
            body.append('\n' + line)
        elif b[0] == 'para': body.append(to_tex(b[1]) + '\n')
        elif b[0] == 'ul': body.append('\\begin{itemize}\n' + '\n'.join(r'\item ' + to_tex(x) for x in b[1]) + '\n\\end{itemize}\n')
        elif b[0] == 'ol': body.append('\\begin{enumerate}\n' + '\n'.join(r'\item ' + to_tex(x) for x in b[1]) + '\n\\end{enumerate}\n')
        else:
            name, head, lines = b[1].split()[0], b[1], b[2]
            attrs = parse_attrs(head + ' ' + ' '.join(l for l in lines if not re.match(r'^[DHR]:', l)))
            if name == 'title': title = to_tex(' '.join(lines).strip(), False)
            elif name == 'abstract': abstract = to_tex(' '.join(lines).strip())
            elif name == 'keywords': keywords = to_tex(' '.join(lines).strip(), False).replace(';', ',')
            elif name == 'figure': body.append(tex_figure(attrs) + '\n')
            elif name == 'table': body.append(tex_table(attrs, [(l[0], l[2:].strip()) for l in lines if re.match(r'^[HR]:', l)]) + '\n')
            elif name == 'equation':
                body.append('\\begin{equation}\n\\label{%s}\n%s\n\\end{equation}\n' % (attrs['id'], '\n'.join(lines).strip()))
            elif name == 'code': body.append('{\\footnotesize\n\\begin{verbatim}\n' + '\n'.join(lines).rstrip() + '\n\\end{verbatim}}\n')
    return PREAMBLE.replace('@@TITLE@@', title).replace('@@ABSTRACT@@', abstract).replace('@@KEYWORDS@@', keywords) + '\n'.join(body) + POSTAMBLE

# ------------------------------------------------------------------ Google-Docs (Markdown) emission
def md_table(rows, header_only_rows=None):
    hdr = [r for r in rows if r[0] == 'H'][0][1].split('|'); data = [r[1].split('|') for r in rows if r[0] in ('R', 'D')]
    out = ['| ' + ' | '.join(to_docs(c.strip()) for c in hdr) + ' |', '|' + '---|' * len(hdr)]
    for r in data: out.append('| ' + ' | '.join(to_docs(c.strip()) for c in r) + ' |')
    return '\n'.join(out)

def build_docs():
    out = []; title = ''
    sec = [0, 0, 0]
    for b in BLOCKS:
        if b[0] == 'head':
            lvl = len(b[1].rstrip('*'))
            if b[1].endswith('*'): out.append('\n## ' + b[2].upper() + '\n'); continue
            if lvl == 1: sec[0] += 1; sec[1] = 0; out.append('\n## %s. %s\n' % (roman(sec[0]), b[2].upper()))
            elif lvl == 2: sec[1] += 1; out.append('\n### %s. %s\n' % (chr(64 + sec[1]), b[2]))
        elif b[0] == 'para': out.append(to_docs(b[1]) + '\n')
        elif b[0] == 'ul': out.append('\n'.join('- ' + to_docs(x) for x in b[1]) + '\n')
        elif b[0] == 'ol': out.append('\n'.join('%d. %s' % (n + 1, to_docs(x)) for n, x in enumerate(b[1])) + '\n')
        else:
            name, head, lines = b[1].split()[0], b[1], b[2]
            attrs = parse_attrs(head + ' ' + ' '.join(l for l in lines if not re.match(r'^[DHR]:', l)))
            if name == 'title':
                title = ' '.join(lines).strip()
                out.append('# ' + title + '\n')
                out.append('**[Student Name 1]**, **[Student Name 2]**, **[Student Name 3]**  \nDepartment of [Department], [Institution], [City, Country]  \n[email1@example.edu], [email2@example.edu], [email3@example.edu]\n')
                out.append('*(Replace the placeholders above; the sample template\'s author block layout should be applied in Google Docs.)*\n')
            elif name == 'abstract': out.append('**Abstract—** ' + to_docs(' '.join(lines).strip()) + '\n')
            elif name == 'keywords': out.append('**Index Terms—** ' + to_docs(' '.join(lines).strip().replace(';', ',')) + '\n')
            elif name == 'figure':
                n = LABELS[attrs['id']]; kind = attrs['kind']
                out.append('> **[INSERT FIGURE %s HERE]**' % n)
                if kind == 'placeholder': out.append('> %s' % attrs['hint'])
                else:
                    out.append('> %s' % attrs['docs'])
                    drows = [('D', l[2:].strip()) for l in lines if l.startswith('D:')]
                    if drows: out.append('\n' + md_table([('H', drows[0][1])] + drows[1:]) + '\n')
                out.append('\n*Fig. %s. %s*\n' % (n, to_docs(attrs['caption'])))
            elif name == 'table':
                n = LABELS[attrs['id']]
                out.append('**TABLE %s**  \n**%s**\n' % (n, to_docs(attrs['caption']).rstrip('.').upper() if False else to_docs(attrs['caption'])))
                out.append(md_table([(l[0], l[2:].strip()) for l in lines if re.match(r'^[HR]:', l)]) + '\n')
            elif name == 'equation':
                out.append('%s    **(%s)**\n' % (attrs['text'].replace('_{', '_').replace('}', ''), LABELS[attrs['id']]))
            elif name == 'code': out.append('```\n' + '\n'.join(lines).rstrip() + '\n```\n')
    out.append('\n## REFERENCES\n')
    for k in CITE_ORDER: out.append('[%d] %s  ' % (CITENUM[k], fmt_ref_docs(k)))
    return '\n'.join(out) + '\n'


def fmt_authors_tex(a):
    names = [x.strip() for x in re.split(r'\s+and\s+', a)]
    out, etal = [], False
    for n in names:
        if n == 'others': etal = True; continue
        if ',' in n:
            last, first = [p.strip() for p in n.split(',', 1)]
            initials = ' '.join(p[0] + '.' for p in re.split(r'[\s-]+', first) if p)
            out.append(f'{initials} {last}')
        else: out.append(n.replace('{', '').replace('}', ''))
    if etal: return ', '.join(out) + ', et al.'
    if len(out) == 1: return out[0]
    if len(out) == 2: return f'{out[0]} and {out[1]}'
    return ', '.join(out[:-1]) + ', and ' + out[-1]

def urlify(t): return re.sub(r'(https?://[^\s}]+)', lambda m: r'\url{' + m.group(1) + '}', t)

def fmt_ref_tex(key):
    typ, f = BIBDB[key]; au = fmt_authors_tex(f['author']); t = f['title']; yr = f.get('year', '')
    if typ == 'book':
        return f"{au}, \\textit{{{t}}}, {f.get('edition','')} ed. {f.get('address','')}: {f.get('publisher','')}, {yr}."
    if typ == 'article':
        s = f"{au}, ``{t},'' \\textit{{{f['journal']}}}, vol. {f['volume']}, no. {f['number']}, pp. {f['pages']}, {yr}"
        if 'doi' in f: s += f", doi: {f['doi']}"
        return s + '.'
    if typ == 'inproceedings':
        s = f"{au}, ``{t},'' in \\textit{{{f['booktitle']}}}, {yr}"
        if 'pages' in f: s += f", pp. {f['pages']}"
        if 'note' in f: s += f", {f['note']}"
        return s + '.'
    hp = f.get('howpublished', '')
    if hp.startswith('arXiv'): return f"{au}, ``{t},'' {hp}, {yr}."
    return f"{au}, ``{t}.'' {urlify(hp)}"

def embed_bibliography(tex):
    items = '\n'.join('\\bibitem{%s}\n%s' % (k, fmt_ref_tex(k)) for k in CITE_ORDER)
    block = '\\begin{thebibliography}{%d}\n%s\n\\end{thebibliography}' % (len(CITE_ORDER), items)
    return tex.replace('\\bibliographystyle{IEEEtran}\n\\bibliography{references}', '% Bibliography embedded so that this file compiles on its own (no references.bib needed).\n' + block)

# ------------------------------------------------------------------ write
open(os.path.join(ROOT, 'VisionRL_Report_GoogleDocs.md'), 'w', encoding='utf8').write(build_docs())
os.makedirs(os.path.join(ROOT, 'overleaf'), exist_ok=True)
open(os.path.join(ROOT, 'overleaf', 'main.tex'), 'w', encoding='utf8').write(embed_bibliography(build_tex()))
if os.path.exists(os.path.join(ROOT, 'overleaf', 'references.bib')): os.remove(os.path.join(ROOT, 'overleaf', 'references.bib'))
print('built: %d citations in order of first appearance, %d figures, %d tables, %d equations' % (len(CITE_ORDER), fig_n, tab_n, eq_n))
