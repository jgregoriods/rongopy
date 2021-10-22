# rongopy
### Ideas for the decipherment of rongorongo using machine learning and genetic algorithms in Python

Jonas Gregorio de Souza<br/>
[![ORCiD](https://img.shields.io/badge/ORCiD-0000--0001--7879--4531-green.svg)](https://orcid.org/0000-0001-6032-4443)<br/>

## Background
### What is rongorongo?
<img src="img/key.png" align="left">
<p>Rongorongo (henceforth RoR) is an undeciphered glyph system from Easter Island. The very nature of RoR as true writing is debated. In the past, the prevalent view was that the glyphs were a mnemonic device and were unrelated to the specific words of the chants they were meant to recall (Métraux 1957; Routledge 1919). Nowadays, most scholars assume that the system was either logographic, with a few phonetic complements (<a href="https://doi.org/10.3406/jso.1990.2882">Guy 1990,</a> <a href="https://kahualike.manoa.hawaii.edu/rnj/vol20/iss1/9/">2006</a>; <a href="https://www.jstor.org/stable/20706625">Fischer 1995a</a>), or predominantly syllabic, with certain glyphs working as determinatives or logograms (<a href="https://doi.org/10.3406/jso.1996.1995">Pozdniakov 1996</a>; <a href="http://pozdniakov.free.fr/publications/2007_Rapanui_Writing_and_the_Rapanui_Language.pdf">Pozdniakov and Pozdniakov 2007</a>; <a href="https://kahualike.manoa.hawaii.edu/rnj/vol19/iss2/6/">Horley 2005,</a><a href="https://kahualike.manoa.hawaii.edu/rnj/vol21/iss1/7/"> 2007</a>).</p>

<p>The canonical RoR corpus is comprised of texts carved on 20 wooden tablets, one staff, two <i>reimiro</i> (pectoral adornments), one birdman sculpture (<i>tagata manu</i>), and one snuffbox (assembled from an earlier tablet). A bark-cloth fragment has recently been recognized as another genuine inscription (<a href="ps://www.sav.sk/index.php?lang=sk&doc=journal-list&part=article_response_page&journal_article_no=17609">Schoch and Melka 2019</a>). The texts are repetitive, with three tablets (H, P, Q) containing the same text. Certain sequences of glyphs, some of them quite long, appear in multiple artefacts.<p>
<p>The only RoR passage whose meaning is thought to be understood by most scholars is the lunar calendar on tablet <i>Mamari</i> (<a href="https://doi.org/10.3406/jso.1990.2882">Guy 1990</a>; <a href="https://doi.org/10.4000/jso.6314">Horley 2011</a>; but see <a href="https://doi.org/10.15286/jps.121.3.243-274">Davletshin 2012b</a>). Repeated crescent-shaped glyphs are combined with other signs, presumably phonetic complements used to spell the names of the nights.</p>
<p>The antiquity of the system is another point of contention (<a href="http://www.jstor.org/stable/20706648">Langdon and Fischer 1996</a>). Most of the artefacts appear to be recent. Three tablets were carved on European oars, and the only radiocarbon measurement available (for tablet Q, Small St. Petersburg) points to the 19th century (<a href="https://doi.org/10.1002/j.1834-4453.2005.tb00597.x">Orliac 2005</a>). If, however, RoR can be proven earlier than the European encounter and its function as real writing can be ascertained, this would be a remarkable finding - one of the rare cases of independent invention of writing in the world.</p>
<p>On this website and <a href="https://github.com/jgregoriods/rongopy">repository</a>, I offer some thoughts on a machine-learning approach towards decipherment, alongside the data (RoR corpus that can be loaded in Python) and code. This is not another claim to decipherment, as I don't think the results were satisfactory, but the method is promising and can perhaps inspire others.</p>

## Approaches to decipherment
<p>The earliest attempts at decipherment, still in the 19th century, took advantage of the fact that informants were still alive who had presumably been instructed in RoR - or at least heard the tablets being recited (Routledge 1919). Two informants, named Metoro and Ure Vaeiko, provided readings for entire tablets (Thompson 1889; Jaussen 1893). Metoro's readings - apparently just a description of the objects depicted by individual glyphs - formed the basis for Thomas Barthel's interpretation of RoR (Barthel 1958).</p>
<p>Yuri Knorozov, famous for the decipherment of Maya glyphs, was later involved with other Soviet scholars in the study of RoR (Butinov and Knorozov 1957). Their understanding was that RoR was a mixed writing with logograms and phonetic complements, similar to other hieroglyphic systems.</p>
<p>The many publications of Jacques Guy opened several routes to decipherment. Most importantly, we must mention the recognition of potential taxograms or determinatives (<a href="https://kahualike.manoa.hawaii.edu/rnj/vol20/iss1/9/">Guy 2006</a>) and the interpretation of the structure of the lunar calendar in tablet <i>Mamari</i>, including a number of plausible phonetic readings for signs that accompany the moon glyphs (<a href="https://doi.org/10.3406/jso.1990.2882">Guy 1990</a>).</p>
<p>In the 1990s, Steven R. Fischer brought renewed attention to the field with his purported decipherment. Based on similarities with the structure of a cosmogonic chant recited by Ure Vaeiko, Fischer read a series of procreation triads in the Santiago Staff (<a href="https://kahualike.manoa.hawaii.edu/rnj/vol9/iss4/1/">Fischer 1995a</a>) and other tablets (<a href="https://www.jstor.org/stable/20706625">Fischer 1995b</a>). His work, however, was heavily criticized by other RoR scholars (<a href="https://doi.org/10.3406/jso.1998.2041">Guy 1998</a>; <a href="https://doi.org/10.3406/jso.1996.1995">Pozdniakov 1996</a>).</p>
<p>The recent work by the Pozdniakovs (<a href="https://doi.org/10.3406/jso.1996.1995">Pozdniakov 1996</a>; <a href="http://pozdniakov.free.fr/publications/2007_Rapanui_Writing_and_the_Rapanui_Language.pdf">Pozdniakov and Pozdniakov 2007</a>) and Paul Horley (<a href="https://kahualike.manoa.hawaii.edu/rnj/vol19/iss2/6/">2005</a>, <a href="https://kahualike.manoa.hawaii.edu/rnj/vol21/iss1/7/">2007</a>) is focused on simplifying Barthel's catalogue by isolating the basic glyphs in RoR and comparing glyph and Rapanui syllable statistics. Similarly, Albert Davletshin (<a href="https://doi.org/10.4000/jso.6658">Davletshin 2012a</a>, <a href="https://doi.org/10.15286/jps.121.3.243-274">2012b</a>) has been attempting to separate syllabograms and logograms in RoR based on glyph combinatorial properties.</p>
<p>Finally, Martyn Harris and Tomi Melka have been moving the field in the direction of machine learning and natural language processing with  <i>n</i>-gram collocation and latent semantic analysis (LSA) (<a href="https://doi.org/10.1080/09296174.2011.556003">Harris and Melka 2011a</a>, <a href="https://doi.org/10.1080/09296174.2011.581850">2011b</a>).</p>

## Revising the glyph catalogue
<p>A sound statistical analysis of RoR depends on the correct transliteration of the texts. Unfortunately, the system devised by Barthel (1958) exaggerates the quantity of glyphs by assigning different numbers to allographs and ligatures. In fact, a quick experiment showed me that, after differentiating the most obvious allographs and separating the most obvious ligatures (e.g. anthropomorphs and ornitomorphs with various hand shapes), we are left with about 50 glyphs accounting for over 90% of the corpus - a number surprisingly close to the number of Rapanui syllables.</p>
<p>Martha Macri (1996), the Pozdniakovs (<a href="http://pozdniakov.free.fr/publications/2007_Rapanui_Writing_and_the_Rapanui_Language.pdf">2007</a>) and Paul Horley (<a href="https://kahualike.manoa.hawaii.edu/rnj/vol19/iss2/6/">2005</a>) have all attempted to simplify Barthel's catalogue, arriving at pretty similar solutions. Horley offers the most radical restructuring, e.g. considering the anthropomorphic and ornitomorphic glyphs' heads as independent signs. For creating the "simplified" corpus, I mostly adopted the Pozdniakovs' solution, except for the treatment of glyphs like those in the series 420-430, which Pozdniakov (<a href="https://doi.org/10.4000/jso.6371">2011</a>) initially regarded as a ligature of hand glyphs 006 or 010 with anthropomorphic or ornitomorphic glyphs.</p>
<p>In fact, it is unclear how those glyphs should be treated. In a recent paper, Pozdniakov (<a href="http://pozdniakov.free.fr/publications/2016_Correlation_of_graphical_features.pdf">2016</a>) cast doubt on whether glyphs of the series 220/240/320/340 should be considered as independent glyphs rather than allographs of 200/300, based on the observation that leg shapes are not independent of hand shapes.</p>
<p>There are reasons to take that argument one step further and consider all anthropomorphic glyphs as allographs. The parallel passages below are striking:</p>
<img src="img/parallels.png" width=400>
<p>Other examples can be found in Aa7:Ra3, Br9:Bv3, Bv3:Ra4, Bv8:Sa5, Bv7 and probably many other places.</p>
<p>If anthropomorphic glyphs in standing (200/300), seating in profile (280/380) and seating in frontal view (240/340) position are allographs, should we view ornitomorphic glyphs in the same manner? For example,  should we treat glyphs in the 430/630 series as "profile" versions of the frontal ornitomorphs (400/600)? Pozdniakov (<a href="http://pozdniakov.free.fr/publications/2016_Correlation_of_graphical_features.pdf">2016</a>) seems to hint at that possibility for glyphs in the 430 series, even suggesting that Barthel (1958) probably thought so. Here, I have adopted that view, merging all anthropomorphs and most ornitomorphs (except for those in the 660 series) in the simplified corpus.</p>

## Usage
### Data exploration

<p>We start by loading the tablets and language corpus.</p>
<p>The following artefacts were retained for the analysis: A, B, C, D, E, G, N, P, R and S. G was selected as inclusive of the text in K, and P was selected as representative of H-P-Q. The Santiago Staff (I) reflects a very particular genre and structure (also present in parts of G-K), and was left out of the analysis for now. The selection can be changed in the <code>config.py</code> file.</p>
<p>The rongorongo corpus is provided as a dictionary with artefacts' names (letters) as keys. Values are themselves dictionaries with each line as key and a string of glyphs as value:</p>

```python
>>> from utils import load_data
>>> from config import TABLET_SUBSET
>>> 
>>> all_tablets = load_data('./tablets/tablets_clean.json')
>>> tablets = {tablet: all_tablets[tablet] for tablet in TABLET_SUBSET}
>>> 
>>> tablets['B']['Br1']
'595-001-050-394-004-002-595-001-050-301-004-002-040-211-091-200-595-002-394-004-002-595-002-050-394-004-002-595-002-050-301-004-002-042-211-091-595-600-050-381-004-002-306-325-430-053-430-017-430-004-002-208-200-002-022-305-074-095-001-000-069'
```

<p>Once a particular set of texts is loaded, basic properties of the glyphs can be explored with the <code>GlyphStats</code> class. For example, the <code>get_percentages()</code> method returns a data frame with the ordered percentages and cumulative percentages of each glyph:</p>

```python
>>> from explore.glyph_stats import GlyphStats
>>> 
>>> glyph_stats = GlyphStats(tablets)
>>> glyph_frequencies = glyph_stats.get_percentages()
>>> glyph_frequencies
    Glyph                 Percent   Cumulative Percent
0     001     0.06461710452766908  0.06461710452766908
1     002     0.03666852990497484  0.10128563443264393
2     004     0.03566238121855785   0.1369480156512018
3     003     0.03476802683063164  0.17171604248183342
4     022     0.02615986584684181  0.19787590832867524
..    ...                     ...                  ...
541   682  0.00011179429849077697   0.9995528228060321
542   555  0.00011179429849077697   0.9996646171045228
543   576  0.00011179429849077697   0.9997764114030135
544   587  0.00011179429849077697   0.9998882057015043
545   635  0.00011179429849077697    0.999999999999995

[546 rows x 3 columns]
```

<p>We see that the top 50 glyphs account for approximately 65% of the texts:</p>

```python
>>> glyph_frequencies.loc[[50]]
   Glyph                Percent  Cumulative Percent
50   711  0.0051425377305757405  0.6471771939631079
```

<p> Let's save a list with the top 50 glyphs to be used later in the genetic algorithm. This can be done with the <code>get_top_n()</code> method:</p>

```python
>>> top_glyphs = glyph_stats.get_top_n(50)
```

<p>Similarly, language properties can be analysed with the <code>LangStats</code> class. The Rapanui language corpus has been compiled ...</p>

```python
>>> from explore.lang_stats import LangStats
>>> 
>>> raw_corpus = load_data('./language/corpus.txt')
>>> lang_stats = LangStats(raw_corpus)
```

