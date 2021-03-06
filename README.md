# rongopy
### Ideas for the decipherment of rongorongo using machine learning and genetic algorithms in Python

Jonas Gregorio de Souza<br/>
[![ORCiD](https://img.shields.io/badge/ORCiD-0000--0001--7879--4531-green.svg)](https://orcid.org/0000-0001-6032-4443)<br/>

## What is rongorongo?
<img src="img/key.png" align="left">
<p>Rongorongo (henceforth RoR) is an undeciphered glyph system from Easter Island. The very nature of RoR as true writing is debated. In the past, the prevalent view was that the glyphs were a mnemonic device and were unrelated to the specific words of the chants they were meant to recall (Métraux 1957; Routledge 1919). Nowadays, most scholars assume that the system was either logographic, with a few phonetic complements (<a href="https://doi.org/10.3406/jso.1990.2882">Guy 1990,</a> <a href="https://kahualike.manoa.hawaii.edu/rnj/vol20/iss1/9/">2006</a>; <a href="https://www.jstor.org/stable/20706625">Fischer 1995a</a>), or predominantly syllabic, with certain glyphs working as determinatives or logograms (<a href="https://doi.org/10.3406/jso.1996.1995">Pozdniakov 1996</a>; <a href="http://pozdniakov.free.fr/publications/2007_Rapanui_Writing_and_the_Rapanui_Language.pdf">Pozdniakov and Pozdniakov 2007</a>; <a href="https://kahualike.manoa.hawaii.edu/rnj/vol19/iss2/6/">Horley 2005,</a><a href="https://kahualike.manoa.hawaii.edu/rnj/vol21/iss1/7/"> 2007</a>).</p>
<p><b>This is not yet another pseudo-decipherment.</b> The RoR literature is already saturated with extraordinary claims. The contents of this repository are meant to illustrate an approach that I was pursuing and that I thought promising. The basic idea is to select mappings of glyphs to syllables through a genetic algorithm. The decoded texts are evaluated by LinearSVC and LSTM models trained on a real Rapanui language corpus to assess how Rapanui-like they are.</p>
<p>Details can be found on the project's <a href="https://jgregoriods.github.io/rongopy/">website</a>.
