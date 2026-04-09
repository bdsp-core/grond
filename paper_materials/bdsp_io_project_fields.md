# bdsp.io project — field-by-field paste sheet

> **What this is.** A copy-paste cheat sheet for filling out the bdsp.io
> ActiveProject edit form for the project you started at:
>
> https://bdsp.io/projects/s58yrgqds7xqn8622x7u/overview/
>
> Open the edit page in a browser, then for each section below open the
> matching field on bdsp.io and paste the suggested content. Most of the
> long-form sections are written as **HTML** because bdsp.io stores them
> as `SafeHTMLField` (CKEditor); paste via the editor's **Source** view
> for cleanest results, or paste the plain rendered text into the
> WYSIWYG view if you prefer.
>
> **Resource type assumption.** I am writing this assuming the project
> resource type is **software** (algorithms + code), since the headline
> deliverable is the GROND characterization system. The Software
> template's section labels are: Abstract / Background / Software
> Description / Technical Implementation / Installation and Requirements
> / Usage Notes / (Release Notes) / Ethics / Acknowledgements /
> Conflicts of Interest / References. If you instead chose **database**
> when starting the project, the section labels are slightly different
> (Methods, Data Description, no Installation) but the actual model
> fields underneath are the same — every section below maps to the
> field name in parentheses, so you can find the right slot regardless.
> If you'd rather change the resource type to database, the
> "Software Description" content below should go in the "Data
> Description" field instead, and "Technical Implementation" should be
> moved up into the Methods field.

---

## Title  *(model field: `title`, max 200 characters)*

```text
Automated Characterization of Periodic and Rhythmic EEG Patterns with GROND: a Generalized Rhythmic and Oscillatory Neurophysiology Descriptor
```

## Short description  *(field: `short_description`, max 250 characters)*

```text
GROND is an automated system for joint characterization of periodic discharges (LPD, GPD, BIPD) and rhythmic delta activity (LRDA, GRDA) in continuous EEG, providing lateralization, spatial localization, frequency, and individual discharge timing.
```

(248 characters — fits.)

## Version  *(field: `version`)*

```text
1.0.0
```

## Project home page  *(field: `project_home_page`)*

```text
https://github.com/bdsp-core/pd-rda-profiler
```

## Programming languages  *(field: `programming_languages`, multi-select)*

- **Python** (only)

## License  *(field: `license`)*

The repo's [LICENSE.txt](../LICENSE.txt) is **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0). Pick the matching entry from the bdsp.io License dropdown.

## Access policy  *(field: `access_policy`)*

**Credentialed** (the bdsp.io default; matches `bdsp-opendata-credentialed` bucket policy).

## DUA  *(field: `dua`)*

Pick the standard BDSP DUA from the dropdown — same as the predecessor Tăuțan et al. project, if you can find it.

## Required trainings  *(field: `required_trainings`)*

Pick the standard BDSP credentialing trainings (CITI Human Subjects Research, etc.) — same set as other BDSP credentialed projects.

---

## Abstract  *(field: `abstract`, max 10000 characters; HTML)*

```html
<p><strong>Objective.</strong> Periodic discharges (PDs) and rhythmic delta activity (RDA) are common electroencephalographic (EEG) patterns in critically ill patients that require detailed characterization&mdash;including lateralization, spatial localization, discharge timing, and frequency estimation&mdash;according to the American Clinical Neurophysiology Society (ACNS) 2021 standardized terminology. Manual characterization is subjective, time-consuming, and exhibits substantial inter-rater variability. This study presents <strong>GROND</strong> (Generalized Rhythmic and Oscillatory Neurophysiology Descriptor), a comprehensive automated system for characterizing all four major subtypes: lateralized periodic discharges (LPD), generalized periodic discharges (GPD), lateralized rhythmic delta activity (LRDA), and generalized rhythmic delta activity (GRDA).</p>

<p><strong>Approach.</strong> GROND comprises two complementary pipelines: the PD-Profiler and the RDA-Profiler. The PD-Profiler combines a per-channel convolutional neural network (ChannelPD-Net) with hemisphere-specific learned evidence traces (HemiCET-UNet), dynamic programming with an approximately-periodic prior, and discharge-locked topographic localization. The RDA-Profiler uses iterative narrowband Hilbert refinement (NB-Hilbert) for frequency and lateralization, with phase-locking value (PLV) analysis for spatial extent. Both pipelines were trained and evaluated on 12,238 EEG segments from 11,578 unique patients, annotated by four expert electroencephalographers through iterative human-in-the-loop label refinement.</p>

<p><strong>Main results.</strong> The system matched or exceeded expert inter-rater agreement on every characterization task. For RDA frequency estimation, the algorithm&ndash;expert intraclass correlation coefficient (ICC) of 0.860 slightly exceeded the expert&ndash;expert ICC of 0.852&mdash;the first automated method to match expert-level agreement for any continuous EEG-pattern attribute. PD discharge timing achieved an F1 of 0.889 at single-sample precision (1.0&nbsp;ms median timing error at 200&nbsp;Hz). PD hemisphere lateralization reached AUC&nbsp;0.989 (n&nbsp;=&nbsp;1,274), and LPD versus GPD classification reached AUC&nbsp;0.911 (n&nbsp;=&nbsp;7,037). RDA lateralization reached AUC&nbsp;0.837 (n&nbsp;=&nbsp;4,253). Frequency estimation Spearman correlations were 0.786 (LPD) and 0.846 (GPD), three to five times higher than the prior signal-processing state of the art. Spatial localization reached 97.3% of expert inter-rater agreement for PDs (Jaccard 0.731 versus expert&ndash;expert 0.751) and matched expert&ndash;expert ICC for RDA (0.371 versus 0.373).</p>

<p><strong>Significance.</strong> GROND is the first system to jointly characterize lateralization, spatial localization, discharge timing, and frequency for both periodic and rhythmic EEG patterns, achieving algorithm&ndash;expert agreement at the level of expert&ndash;expert agreement across all tasks. In post-hoc review of discordant cases, experts judged the algorithm's frequency estimates more accurate than the original expert labels in 94% of cases. Automated characterization can now both substitute for and improve manual annotation in critical-care EEG.</p>
```

---

## Background  *(field: `background`; HTML)*

```html
<p>Continuous EEG (cEEG) monitoring is widely used across acute-care settings for detecting seizures and identifying patterns associated with secondary brain injury in critically ill patients. Among the most clinically significant findings are periodic discharges (PDs) and rhythmic delta activity (RDA), which lie along the ictal&ndash;interictal continuum and are associated with increased risk of seizures, neuronal injury, and poor outcomes. The ACNS standardized critical care EEG terminology, revised in 2021, provides a framework for describing these patterns along multiple dimensions: main term (periodic vs. rhythmic), lateralization (lateralized vs. generalized vs. bilateral independent), spatial distribution, and quantitative features including repetition frequency and regularity.</p>

<p>A growing body of evidence indicates that these patterns are not merely markers of brain injury but can themselves contribute to neuronal damage. Microdialysis and FDG-PET studies have shown that periodic discharges, like seizures, are accompanied by metabolic crisis and elevated glucose utilization in injured cortex. At the population level, our group has shown that the proportion of monitoring time a patient spends with periodic or rhythmic epileptiform activity is independently associated with worse discharge neurological outcomes (Zafar et al., <em>Ann Neurol</em> 2021; Parikh et al., <em>Lancet Digit Health</em> 2023). These results are consistent with a dose-response, plausibly causal contribution of periodic and rhythmic epileptiform activity to neurological injury.</p>

<p>Time-burden alone, however, is a coarse summary of what is fundamentally a multidimensional phenomenon. Both theoretical considerations and prior experimental work suggest that the <em>characteristics</em> of the epileptiform state matter: faster discharge frequencies and broader spatial involvement plausibly impose greater metabolic demand on vulnerable cortex. Until now, automated tools could not reliably quantify these features at the data scales required for population studies. Existing systems either restricted themselves to detection and classification of pattern subtype without quantifying frequency or spatial extent, or attempted these measurements with signal-processing methods whose agreement with experts was far below expert&ndash;expert agreement. Building tools that can measure frequency, spatial localization, lateralization, and individual discharge timing reproducibly and at scale is the principal motivation for this work.</p>
```

---

## Software description  *(field: `content_description`; HTML)*

> If your project is a **database** instead of software, paste this into the "Data Description" field. If you want to also describe the labeled dataset, append the second block below.

```html
<p>GROND is organized around two complementary pipelines that share a 19-channel monopolar EEG input (10 seconds at 200&nbsp;Hz, common-average-reference montage). Both pipelines emit ACNS 2021-formatted verbal descriptions of the form &ldquo;LPD, left-sided, at 1.5&nbsp;Hz, left frontotemporal predominant.&rdquo;</p>

<h3>PD-Profiler (LPD, GPD, BIPD)</h3>
<ul>
  <li><strong>ChannelPD-Net</strong> &mdash; lightweight per-channel 1D CNN with temporal attention pooling (~50K parameters) that emits per-channel PD probability and log-frequency. Used for hemisphere lateralization (mean PD probability per side) and as a frequency prior.</li>
  <li><strong>HemiCET-UNet</strong> &mdash; hemisphere-restricted convolutional evidence-trace U-Net (~525K parameters) that produces a frame-level discharge evidence trace at 200&nbsp;Hz. Trained on Gaussian-peak targets at expert-annotated discharge times.</li>
  <li><strong>Handcrafted Peak Prior (HPP)</strong> &mdash; pointiness + Teager&ndash;Kaiser energy ratio, combined with the HemiCET evidence via a product-boost formula.</li>
  <li><strong>Dynamic programming</strong> &mdash; forward DP with an approximately-periodic prior recovers the optimal discharge sequence. Three rounds of EM-style template refinement sharpen timing.</li>
  <li><strong>Discharge-locked topographic localization</strong> &mdash; mean voltage topography computed at the moment of each detected discharge after Laplacian transform and GFP-aligned averaging. Bypasses the noisy expert region-counting task by using a direct voltage measurement at known event times.</li>
  <li><strong>BIPD detection</strong> &mdash; gradient-boosted-tree classifier on bilateral timing features (matched fraction, phase consistency, frequency asymmetry). Trained on synthetic BIPD examples (paired LPD sequences) and synthetic GPD examples (phase-shifted single LPD sequence) to overcome the rarity of confirmed BIPD cases.</li>
</ul>

<h3>RDA-Profiler (LRDA, GRDA)</h3>
<ul>
  <li><strong>NB-Hilbert</strong> &mdash; two-pass iterative narrowband Hilbert refinement. Pass&nbsp;1: broadband (0.5&ndash;3.5&nbsp;Hz) variance-based hemisphere lateralization and Hilbert instantaneous-frequency estimation. Pass&nbsp;2: narrowband filter centered on the pass-1 frequency, refined lateralization from envelope amplitude, refined frequency on the dominant hemisphere only.</li>
  <li><strong>RDA-PLV spatial extent</strong> &mdash; per-channel PLV with the dominant-hemisphere reference, multiplied by normalized envelope amplitude. Channels exceeding a calibrated threshold are classified as involved.</li>
</ul>

<h3>Contest of methods</h3>
<p>Each subtask in the pipeline (PD frequency, PD spatial localization, PD discharge timing, RDA lateralization, RDA frequency) was developed through a systematic &ldquo;contest of methods&rdquo; in which more than 300 candidate algorithm variants were implemented and benchmarked before any single approach was chosen. The 76 variants of the V5 RDA lateralization contest, the 26-method PD spatial-localization contest, and 9 refinement rounds of PD frequency estimation are documented in the project's evaluation logs. NB-Hilbert (entry W05) was the overall winner of the RDA contest; HemiCET-UNet&nbsp;+ DP was the winner of the PD discharge-timing contest.</p>
```

**Optional second block — labeled dataset description** (paste this after the above if your project also distributes the labels):

```html
<h3>Labeled dataset</h3>
<p>The training and evaluation set comprises <strong>12,238 ten-second EEG segments</strong> from <strong>11,578 unique patients</strong>, drawn from three sources: (1) the IIIC dataset (3,529 segments with crowd-sourced labels from at least 10 expert raters per segment); (2) pattern-specific clinical EEG databases classified by a single expert rater; and (3) a 38-patient four-rater expert dataset. Per-segment label coverage:</p>
<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>&nbsp;</th><th>LPD</th><th>GPD</th><th>LRDA</th><th>GRDA</th><th>Total</th></tr>
  <tr><td>Segments (non-excluded)</td><td>4,170</td><td>3,337</td><td>1,408</td><td>3,323</td><td>12,238</td></tr>
  <tr><td>Unique patients</td><td>3,953</td><td>3,086</td><td>1,381</td><td>3,159</td><td>11,578</td></tr>
  <tr><td>Expert-reviewed frequency</td><td>1,499</td><td>1,539</td><td>654</td><td>1,381</td><td>5,073</td></tr>
  <tr><td>Discharge timing</td><td>917</td><td>1,036</td><td>189</td><td>313</td><td>2,159</td></tr>
  <tr><td>Spatial annotations</td><td>352</td><td>260</td><td>29</td><td>177</td><td>818</td></tr>
  <tr><td>IIIC crowd votes (&ge;10 raters)</td><td>1,846</td><td>1,024</td><td>239</td><td>420</td><td>3,529</td></tr>
</table>
<p>Only one patient contributed segments to more than one subtype, so the row sum essentially equals the deduplicated total.</p>
```

---

## Methods / Technical Implementation  *(field: `methods`; HTML)*

> In the **software** template this is rendered as "Technical Implementation"; in the **database** template it is "Methods". Same field underneath.

```html
<h3>Dataset and annotations</h3>
<p>Each segment consisted of 10&nbsp;seconds of 19-channel monopolar EEG recorded at 200&nbsp;Hz in common-average-reference montage. Segments were classified into four subtypes based on the dominant pattern: LPD, GPD, LRDA, GRDA. Annotations were collected through three layered tasks: pattern classification + laterality (using a symmetric channel-layout viewer), expert frequency review (via interactive viewers that pre-filled the algorithm's estimate for fast accept/override), and per-discharge timing (three rounds of iterative model-assisted review by an expert annotator).</p>

<h3>PD-Profiler training</h3>
<p>ChannelPD-Net was trained with 5-fold patient-stratified cross-validation using a multi-task loss combining binary cross-entropy (PD probability) and masked MSE (log-frequency). HemiCET-UNet was trained on 675 segments with ground-truth discharge timing, with weighted BCE (positive weight 20), a sharpness penalty, and a floor loss enforcing learned evidence at labeled discharge locations. Augmentations included amplitude scaling (0.8&ndash;1.2&times;), Gaussian noise (20&ndash;40&nbsp;dB SNR), channel dropout (p&nbsp;=&nbsp;0.15), and discharge-time jitter ($\sigma$&nbsp;=&nbsp;5&nbsp;ms). The hemisphere-restricted 8-channel input outperformed the full 18-channel bipolar montage by avoiding contamination from the uninvolved hemisphere in lateralized patterns.</p>

<h3>Dynamic programming for discharge timing</h3>
<p>A forward DP with an approximately-periodic prior detected individual discharge times. Candidate peaks were extracted from the combined (HPP + HemiCET) evidence trace within an automatically detected active interval. The DP scored candidates with superlinear evidence rewards, quadratic penalties for deviations from the expected period (alpha&nbsp;=&nbsp;1.275), and skip penalties allowing up to 3 consecutive missed discharges (beta&nbsp;=&nbsp;0.3 per skip). The optimal path was recovered by backtracking, then refined through three rounds of EM-style template-correlation re-detection. Final frequency was the reciprocal of the median inter-peak interval, which proved more accurate than CNN or autocorrelation-based estimates because it leveraged the actual detected timing sequence.</p>

<h3>RDA-Profiler iterative narrowband refinement</h3>
<p>NB-Hilbert was selected as the best unified (lateralization plus frequency) method from a systematic contest of 76 algorithm variants on 4,253 RDA segments. Pass&nbsp;1 used a broadband 0.5&ndash;3.5&nbsp;Hz Butterworth filter; mean variance per hemisphere gave a coarse lateralization, and Hilbert instantaneous frequency on the top-3 channels of the dominant hemisphere gave a coarse frequency. Pass&nbsp;2 applied a narrowband filter at the coarse-frequency &plusmn;&nbsp;0.4&nbsp;Hz; refined lateralization from envelope amplitude; refined frequency from the Hilbert instantaneous frequency on the dominant hemisphere only. The narrowband filter in pass&nbsp;2 sharpened the lateralization signal by suppressing non-RDA activity.</p>

<h3>Spatial localization</h3>
<p>Two complementary spatial methods are implemented. <em>Hybrid-PLV</em> scores per-channel involvement using a hybrid of CNN probabilities and PLV against the top-3 reference channels, mapped to 8 anatomical regions; channels exceeding a calibrated threshold are classified as involved and the spatial extent is reported as the involved-region fraction. <em>Discharge-locked topographic localization</em> (the recommended primary spatial output for PDs) computes the mean voltage topography at the moment of each detected discharge: the surface-Laplacian-transformed signal is GFP-aligned within &plusmn;&nbsp;25&nbsp;ms of each discharge, and the mean topography is computed as a GFP-squared-weighted average across all refined discharge epochs. The squared weighting strongly suppresses phantom (DP-interpolated but not real) discharges. The resulting 19-electrode topography is rendered with MNE-Python spherical-spline interpolation and mapped to 16 anatomical regions for verbal description.</p>

<h3>Evaluation framework</h3>
<p>All evaluations used patient-stratified cross-validation: 5-fold for trained models (ChannelPD-Net, HemiCET-UNet) and full-dataset evaluation for non-trained methods (NB-Hilbert, RDA-PLV). Inter-rater reliability was quantified using ICC(3,1) of Shrout and Fleiss, with the algorithm treated as an additional rater so algorithm&ndash;expert ICC could be compared directly with expert&ndash;expert ICC. Bootstrap 95% confidence intervals (1,000 iterations) were used for IRR. Quality-filtered frequency evaluation required either an MW expert review, a three-rater agreement, or at least 10 IIIC crowd votes with at least 80% pattern agreement.</p>
```

---

## Installation and Requirements  *(field: `installation`; HTML)*

```html
<p>GROND is implemented in Python 3.11 and requires PyTorch (with optional CUDA or Metal acceleration), scipy, numpy, scikit-learn, MNE-Python, and a few smaller dependencies. The full conda environment is shipped with the repo as <code>code/morgoth.yml</code>. To recreate it:</p>

<pre><code>git clone https://github.com/bdsp-core/pd-rda-profiler.git
cd pd-rda-profiler
conda env create -f code/morgoth.yml
conda activate morgoth</code></pre>

<p>Inference runs on CPU, CUDA, or Apple Silicon (Metal Performance Shaders). The trained model checkpoints are distributed via the BDSP S3 bucket at <code>s3://bdsp-opendata-credentialed/iiic-freq3/</code> and are not included in the GitHub repository directly because of size; the repo's <code>data/</code> tree expects to be populated from S3 (see <a href="https://github.com/bdsp-core/pd-rda-profiler/blob/main/README.md">the README</a> for the directory layout).</p>

<p>A second conda environment, <code>code/environment.yml</code>, provides the historical training-time dependencies (foe-builds-pinned for reproducibility); most users will not need it.</p>
```

---

## Usage Notes  *(field: `usage_notes`; HTML)*

```html
<p>The two pipelines (PD-Profiler, RDA-Profiler) are independent and process 10-second 19-channel monopolar EEG segments at 200&nbsp;Hz. Each emits a structured Python dictionary containing lateralization, spatial-region descriptors, frequency, individual discharge times (PD only), and the ACNS 2021-formatted verbal description string.</p>

<p>End-to-end usage is documented in the repo README and the example scripts under <code>code/</code>; in brief:</p>

<pre><code># Single-segment PD characterization
from code.pd_profiler import PDProfiler
result = PDProfiler().characterize(eeg_19ch_2000samp, subtype='lpd')
print(result['verbal_description'])
# &gt; "LPD, left-sided, at 1.50 Hz, left frontotemporal predominant"

# Single-segment RDA characterization
from code.rda_profiler import RDAProfiler
result = RDAProfiler().characterize(eeg_19ch_2000samp, subtype='lrda')</code></pre>

<p><strong>Reproducing the manuscript figures and tables.</strong> The full set of paper figures and tables can be regenerated in fast mode (using cached intermediate results, ~30&nbsp;sec total) with:</p>

<pre><code>conda run -n morgoth python paper_materials/generate_all_figures.py
conda run -n morgoth python paper_materials/generate_all_tables.py</code></pre>

<p><strong>Integration with Morgoth.</strong> GROND is designed to run downstream of <em>Morgoth</em>, our existing pattern-classification system that separates LPD/GPD/LRDA/GRDA from background and from each other at expert-level performance. Morgoth currently labels EIPDs as GPDs by default; GROND then takes Morgoth's LPD and GPD outputs and further characterizes them, including possible re-classification of borderline cases as BIPD via the BIPD detection module.</p>

<p><strong>Independent expert validation.</strong> An independent rater study is in progress to validate the system against experts not involved in algorithm development. The 200-segment-per-task case selection and labeling viewers used for that study are documented under <code>paper_materials/independent_expert_tasks/</code> in the repo.</p>
```

---

## Ethics statement  *(field: `ethics_statement`; HTML)*

```html
<p>This study was conducted under protocols approved by the institutional review boards of Massachusetts General Hospital (protocols #2023P000487 and #2024P002630) and Beth Israel Deaconess Medical Center (protocols #2022P000481 and #2022P000417). Both review boards waived the requirement for informed consent for this retrospective analysis of de-identified EEG recordings.</p>
```

---

## Acknowledgements  *(field: `acknowledgements`; HTML)*

```html
<p>The authors thank the expert electroencephalographers who annotated frequency, discharge timing, and spatial extent for the segments used in this study.</p>

<p><strong>Funding.</strong> Dr.&nbsp;Westover's laboratory is supported by grants from the National Institutes of Health (R01AG073410, R01HL161253, R01NS126282, R01AG073598, R01NS131347, R01NS130119) and by Amazon Web Services (AWS).</p>
```

---

## Conflicts of interest  *(field: `conflicts_of_interest`; HTML)*

```html
<p>Dr.&nbsp;Westover is a co-founder of, serves as a scientific advisor and consultant to, and has a personal equity interest in Beacon Biosignals. The remaining authors declare no competing interests.</p>
```

---

## References  *(separate Reference table on the edit form, one row per citation)*

Add these one-by-one in the References section of the edit form. The bdsp.io references widget usually takes a single rendered string per row.

1. Hirsch LJ, Fong MWK, Leitinger M, et al. American Clinical Neurophysiology Society's standardized critical care EEG terminology: 2021 version. *J Clin Neurophysiol*. 2021;38(1):1–29.
2. Claassen J, Mayer SA, Kowalski RG, Emerson RG, Hirsch LJ. Detection of electrographic seizures with continuous EEG monitoring in critically ill patients. *Neurology*. 2004;62(10):1743–1748.
3. Chong DJ, Hirsch LJ. Which EEG patterns warrant treatment in the critically ill? Reviewing the evidence for treatment of periodic epileptiform discharges and related patterns. *J Clin Neurophysiol*. 2005;22(2):79–91.
4. Rodriguez Ruiz A, Vlachy J, Lee JW, et al. Association of periodic and rhythmic electroencephalographic patterns with seizures in critically ill patients. *JAMA Neurol*. 2017;74(2):181–188.
5. Vespa P, Tubi M, Claassen J, et al. Metabolic crisis occurs with seizures and periodic discharges after brain trauma. *Ann Neurol*. 2016;79(4):579–590. doi:10.1002/ana.24606.
6. Struck AF, Westover MB, Hall LT, Deck GM, Cole AJ, Rosenthal ES. Metabolic correlates of the ictal–interictal continuum: FDG-PET during continuous EEG. *Neurocrit Care*. 2016;24(3):324–331. doi:10.1007/s12028-016-0245-y.
7. Tao JX, Qin X, Wang Q. Ictal–interictal continuum: a review of recent advancements. *Acta Epileptol*. 2020;2:13. doi:10.1186/s42494-020-00021-1.
8. Zafar SF, Subramaniam T, Osman G, Herlopian A, Struck AF. Electrographic seizures and ictal–interictal continuum (IIC) patterns in critically ill patients. *Epilepsy Behav*. 2020;106:107037. doi:10.1016/j.yebeh.2020.107037.
9. Zafar SF, Rosenthal ES, Jing J, et al. Automated annotation of epileptiform burden and its association with outcomes. *Ann Neurol*. 2021;90(2):300–311. doi:10.1002/ana.26161.
10. Parikh H, Hoffman K, Sun H, et al. Effects of epileptiform activity on discharge outcome in critically ill patients in the USA: a retrospective cross-sectional study. *Lancet Digit Health*. 2023;5(8):e495–e502. doi:10.1016/S2589-7500(23)00088-2.
11. Gaspard N, Hirsch LJ, LaRoche SM, Hahn CD, Westover MB; Critical Care EEG Monitoring Research Consortium. Interrater agreement for critical care EEG terminology. *Epilepsia*. 2014;55(9):1366–1373.
12. Gaspard N, Manganas L, Rampal N, Petroff OAC, Hirsch LJ. Similarity of lateralized rhythmic delta activity to periodic lateralized epileptiform discharges in critically ill patients. *JAMA Neurol*. 2013;70(10):1288–1295. doi:10.1001/jamaneurol.2013.3475.
13. Leitinger M, Beniczky S, Rohracher A, et al. Salzburg consensus criteria for non-convulsive status epilepticus — revisited: a critical reappraisal. *Epilepsy Behav*. 2015;49:158–163.
14. Fürbass F, Hartmann MM, Halford JJ, et al. Automatic detection of rhythmic and periodic patterns in critical care EEG based on American Clinical Neurophysiology Society (ACNS) standardized terminology. *Neurophysiol Clin*. 2015;45(3):203–213. doi:10.1016/j.neucli.2015.08.001.
15. Herta J, Koren J, Fürbass F, et al. Prospective assessment and validation of rhythmic and periodic pattern detection in NeuroTrend: a new approach for screening continuous EEG in the intensive care unit. *Epilepsy Behav*. 2015;49:273–279. doi:10.1016/j.yebeh.2015.04.064.
16. McGraw CM, Rao S, Manjunath S, Jing J, Westover MB. Automated quantification of periodic discharges in human electroencephalogram. *Biomed Phys Eng Express*. 2024;10(6):065003. doi:10.1088/2057-1976/ad7165.
17. Jing J, Sun H, Kim JA, et al. Development of expert-level automated detection of epileptiform discharges during electroencephalogram interpretation. *JAMA Neurol*. 2020;77(1):103–108. doi:10.1001/jamaneurol.2019.3485.
18. Jing J, Ge W, Struck AF, et al. Inter-rater reliability of expert electroencephalographers identifying seizures and rhythmic and periodic patterns in EEGs. *Neurology*. 2023;100(17):e1737–e1749. doi:10.1212/WNL.0000000000207003.
19. Jing J, Ge W, Hong S, et al. Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation. *Neurology*. 2023;100(17):e1750–e1762.
20. Ge W, Jing J, An S, et al. Deep active learning for interictal ictal injury continuum EEG patterns. *J Neurosci Methods*. 2023;390:109835.
21. Zheng WL, Amorim E, Jing J, et al. Predicting neurological outcome from electroencephalogram dynamics in comatose patients after cardiac arrest with deep learning. *IEEE Trans Biomed Eng*. 2022;69(5):1813–1825.
22. Tăuțan AM, Jing J, Basovic L, et al. Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data. *J Neural Eng*. 2025;22(6):066027. doi:10.1088/1741-2552/ae2716.
23. Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python. *Front Neurosci*. 2013;7:267. doi:10.3389/fnins.2013.00267.
24. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychol Bull*. 1979;86(2):420–428. doi:10.1037/0033-2909.86.2.420.
25. Lake BM, Ullman TD, Tenenbaum JB, Gershman SJ. Building machines that learn and think like people. *Behav Brain Sci*. 2017;40:e253.

---

## Authors  *(separate Authors panel on the edit form)*

The bdsp.io author list should match the manuscript byline. From [paper_materials/manuscript.tex](manuscript.tex):

| Order | Name | Affiliation | Role |
|---|---|---|---|
| 1  | Jin Jing                    | BIDMC / Harvard Medical School | co-first author |
| 2  | ChenXi Sun                  | BIDMC / Harvard Medical School / Stanford | co-first author |
| 3  | Tianyu Zhang                | BIDMC / Harvard Medical School |  |
| 4  | Matthew Byrd                | BIDMC / Harvard Medical School |  |
| 5  | Alexandra-Maria Tăuțan      | MGH / Harvard Medical School |  |
| 6  | Lara Basovic                | MGH / Harvard Medical School | annotation |
| 7  | Peter N Hadar               | MGH / Harvard Medical School | annotation |
| 8  | Marta P Fernandes           | MGH / Harvard Medical School |  |
| 9  | Daniel Goldenholz           | BIDMC / Harvard Medical School |  |
| 10 | Jennifer Kim                | Yale School of Medicine |  |
| 11 | Aaron F Struck              | Washington University in St. Louis |  |
| 12 | Sahar F Zafar               | MGH / Harvard Medical School | co-senior author, annotation |
| 13 | M Brandon Westover          | BIDMC / Harvard Medical School / Stanford | **corresponding, co-senior, submitting** |

The corresponding-author email is **mb.westover@gmail.com** (note: the manuscript previously had `mwestover@mgh.harvard.edu`, which was updated to `mbwest@stanford.edu` in the latest commit; pick whichever you actually want on the bdsp.io project page).

> The bdsp.io author panel asks each co-author to log in and confirm their authorship. The most reliable workflow is to enter each co-author by their bdsp.io account username (or invite them by email if they don't have an account yet). The four colleagues currently doing the independent expert annotation tasks (Peter Hadar, Lara Basovic, Sahar Zafar, plus the three new colleagues you haven't yet added) should be confirmed as authors before publishing.

---

## Parent project  *(field: `parent_projects`)*

Link to the predecessor BDSP project for the Tăuțan et al. *J Neural Eng* 2025 paper, if it has been published on bdsp.io:

> Tăuțan AM, Jing J, Basovic L, et al. *Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data.* J Neural Eng 22(6):066027 (2025). doi:10.1088/1741-2552/ae2716

If the predecessor isn't on bdsp.io yet, leave this field empty.

---

## Files  *(separate Files panel on the edit form)*

The Files section is where you upload (or link via S3 manifest) the actual project payload. Suggested layout to mirror the GitHub repo structure:

```
/code/                    Python source for PDProfiler, RDAProfiler, contest framework
/data/labels/             segment_labels.csv, annotations.csv, discharge_times.json,
                          channel_involvement.json, rda_wave_labels.json
/data/eeg/                12,238 ten-second .mat files (200 Hz, 19 channels) — already on
                          S3 at s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/
/data/dl_cache/           pre-computed model checkpoints (ChannelPD-Net, HemiCET-UNet)
/paper_materials/         manuscript.tex, figures/, tables/, generate_all_figures.py
README.md
LICENSE.txt
```

The S3 bucket already hosts most of this content under the `iiic-freq3/` prefix, so the bdsp.io manifest workflow should be able to register the existing objects rather than re-upload them. If you want, I can prepare an explicit manifest CSV mapping S3 keys to project file paths — say the word.

---

## After you save

A few things worth doing once the form is filled in:

1. **Preview** the project page from the edit form's "Preview" link. Verify the abstract renders, the references are numbered correctly, and the table in the Software/Data Description section displays.
2. **Invite the co-authors** via the Authors panel and ask them to confirm.
3. **Pick the DUA + required trainings** from the dropdowns to match the predecessor Tăuțan project.
4. **Submit for review** when you're ready. The bdsp.io editors will assign DOIs at publication time (DOI registration is disabled in development; lives in production only).

If anything in this cheat sheet is wrong (the resource type, the author order, the funding list, etc.) just tell me and I'll regenerate the affected sections.
