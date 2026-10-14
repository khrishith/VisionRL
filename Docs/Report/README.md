# VisionRL report: how to use the two deliverables

| File | What it is |
|---|---|
| `VisionRL_Report_GoogleDocs.md` | **Deliverable 1.** Full report text for Google Docs (headings, tables, figure placeholders, numbered references). |
| `overleaf/main.tex` + `overleaf/references.bib` | **Deliverable 2.** Complete IEEEtran conference paper. |
| `VisionRL_overleaf.zip` | The two Overleaf files zipped, for *New Project > Upload Project*. |
| `tools/paper_source.md`, `tools/build_report.py`, `tools/validate_report.py` | The single source both versions are generated from, and the static checker. Edit the source, then run `python3 tools/build_report.py` to regenerate both. |

**Template caveat.** The college template (`Paper Template (1).docx`) could not be opened from here (macOS blocked read access to Downloads), so both versions follow the standard IEEE conference layout. Please compare them with your template (section names, font sizes, margins, reference-count rule) and tell me about any difference.

## A. Transfer to Google Docs

1. Open your college template in Google Docs (File > Import / Open) and *Make a copy*. Keep its title block, margins, fonts and two-column section settings.
2. In the new copy, enable **Tools > Preferences > Enable Markdown**, then open `VisionRL_Report_GoogleDocs.md`, select all, copy, and use **Edit > Paste from Markdown**. (If you prefer plain paste, headings and tables will need re-formatting by hand.)
3. Replace the author placeholders: `[Student Name 1]`, `[Student Name 2 / Guide Name]`, `[Department]`, `[Institution]`, `[City, Country]`, emails. Delete the italic note under the author block.
4. Apply the template's styles: title, author block, *Abstract*/*Index Terms* paragraph, level-1 headings (`## I. ...`, template numbering style), level-2 headings (`### A. ...`), body text, caption style (figure captions below figures, table captions above tables).
5. Apply the two-column layout to the body (not the title block and abstract, if the template keeps those single-column): Format > Columns > 2.
6. Insert the 8 figures at the `[INSERT FIGURE n HERE]` markers, then delete the grey note lines:

   | Fig. | Insert |
   |---|---|
   | 1 | Draw the architecture diagram (instructions in the note; Insert > Drawing) |
   | 2 | Photo/screenshot: `Reports/stage4-room-overview.png` (or a new Scene-view screenshot of `EscapeRoomRandomExit.unity`) |
   | 3 | `Reports/stage3-robot-front.png` |
   | 4 | `Reports/stage8-camera-exits-montage.png` |
   | 5 | Draw the workflow chart (instructions in the note) |
   | 6 | `Reports/stage8-diag-frames-19-26.png` |
   | 7 | Line chart from the small data table under the marker (Insert > Chart) |
   | 8 | Column chart from the data table under the marker (Insert > Chart) |

   Delete the helper data tables once the charts exist, or keep them if the template allows.
7. Check that table captions are above tables and use the template's "TABLE I" style, and that the equation number (1) is right-aligned.
8. Reference list: keep the numbers as they are (they follow order of first citation). Re-apply the template's reference style (IEEE style is already used).
9. Final pass: spell-check, update the field-of-study/acknowledgment text, and check page count (see "Length" below).

## B. Compile in Overleaf

1. Overleaf: **New Project > Upload Project**, choose `VisionRL_overleaf.zip` (or **Blank Project**, then paste `main.tex` and create `references.bib` and paste its content).
2. Menu > Compiler: **pdfLaTeX** (default). Main document: `main.tex`. Compile twice if references or numbers look like `?`.
3. No image files are needed. Figures 1, 5, 7 and 8 are drawn in LaTeX (TikZ/pgfplots); figures 2, 3, 4, 6 are boxed placeholders. To insert a real screenshot: upload it to a `figures/` folder in Overleaf, then in `main.tex` replace the `\figplaceholder{...}{...}{...}` line with the `\includegraphics` line printed in the comment just above it.
4. Replace the author placeholders in the `\author{...}` block (delete the second `\and` block if there is one author).
5. If your college provides its own `.cls` or required packages, add them and adjust the preamble.

## C. What still needs your input

- Names, department, institution, city, emails, acknowledgment text, supervisor name.
- The repository/URL placeholder in Section VII ("repository URL to be added by the authors").
- Real screenshots for Figs. 2, 3, 4, 6 (files listed above, all inside this project's `Reports/` folder).
- Please confirm these facts that came from the project's own files and logs, not from your message: the PPO hyperparameters (Table III), the reward values (Table I), the 48-action limit and 6-step minimum start distance, the software versions (Table IV), the 27.6-minute training time, the fidelity measurements for the *faulty* player (Table V, left column), the run-01 results, and the preliminary fixed-exit results (Section VI-F).
- Reference check (see below).

## D. References: verification status

- Checked against live sources during this work: Brown, Cai and DasGupta (*Statistical Science* 16(2):101-133, doi 10.1214/ss/1009213286); Ng, Harada and Russell (ICML 1999, pp. 278-287); Wilson (*JASA* 22(158):209-212; DOI deliberately omitted because only a third-party listing was found); the ML-Agents GitHub repository exists at the URL given.
- All other entries are well-known works whose titles, authors, venues and arXiv numbers are given from memory. Before submission, open `https://arxiv.org/abs/<number>` for each `arXiv:` entry and compare the title and authors; the numbers are:
  1707.06347, 1502.05477, 1506.02438, 1602.01783, 1312.5602, 2004.13649, 2004.04136, 1609.05143, 1611.03673, 1702.03920, 1904.01201, 1911.00357, 1712.05474, 1612.03801, 1605.02097, 1807.06757, 1606.01540, 1809.02627, 1912.01703, 1812.02341, 1912.01588, 1703.06907, 1709.06560, 2005.12729, 2006.05990, 2108.13264.
- The Nature DQN paper (518(7540):529-533, doi 10.1038/nature14236) and Sutton and Barto (2nd ed., MIT Press, 2018) should also be checked.
- Entries written with "and others" list only the first authors (BibTeX prints "et al.").
- All 34 references are cited in the text; the college's minimum of 23 is exceeded.

## E. Length

The report is long for a mini-project (about 8,000 words including references; roughly 10-11 two-column pages with 8 figures and 8 tables). To shorten it: drop Table II (reward audit) and Fig. 3 (robot), merge Tables VII and VIII, and shorten Section II-C/II-E. Tell me the page limit and I will trim it.

## F. Static validation (run `python3 tools/validate_report.py`)

Checks: ASCII only; balanced braces; matching `\begin/\end`; every `\cite` key in the `.bib` and every `.bib` entry cited; every `\ref` has a `\label`; every figure/table referenced in the text; no unescaped `%`/`#`; no leftover markup; key result numbers present in both versions; citation numbers in increasing first-use order; 34 references in both versions. **LaTeX was not compiled here** (no LaTeX installation on this machine), so Overleaf's first compile is the real test. Any warning such as "Overfull hbox" is cosmetic.
