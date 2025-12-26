# Biosafety-Oriented Sequence Novelty Screening with a Variational Autoencoder

## 1. Abstract

This project implements a biosafety-focused pipeline for screening biological sequences for potential novelty and risk signals. A variational autoencoder (VAE) is trained on a baseline of sequences representing known safe proteins and sequences associated with antimicrobial resistance (AMR) and virulence factors (VF). At inference time, new sequences are embedded into a hand-crafted feature space combining k-mer statistics and physicochemical descriptors, and reconstruction error from the VAE is used as a simple novelty / anomaly score.

The goal is **not** to provide an operational, production-ready biosafety filter, but to prototype and illustrate how unsupervised modeling and domain-motivated feature engineering can complement rule-based screening approaches. The code is intentionally lightweight and transparent so that its behaviour can be inspected and audited.

---

## 2. Biosafety Context and Scope

- **Intended use.** Exploratory research / educational tool for understanding how generative models can detect unusual protein sequences relative to a fixed baseline.
- **Non-intended use.** This pipeline must **not** be used as a sole decision-maker for approving or rejecting DNA/protein synthesis requests, nor as a standalone biosafety control.
- **Threat model.** The system is designed to flag sequences that differ substantially from a baseline of safe and known-threat sequences, under the assumption that strong deviations in feature space may correspond to novel or engineered functions.
- **Limitations.**
  - Reconstruction error is a very coarse signal; high error does not necessarily imply biological threat.
  - Training data are limited and may be biased; the model does not cover the full space of safe or harmful sequences.
  - There is no explicit functional prediction (e.g., toxin activity, host range) and no wet-lab validation.

You should therefore treat this model as an **auxiliary tool** that might highlight sequences for further expert review, not as a replacement for existing biosafety screening standards (e.g. IGSC guidelines, internal red-team processes).

---

## 3. Data and Preprocessing

### 3.1 Datasets

The project expects three FASTA files under `data/`:

- `safe_baseline.fasta` — sequences representing a background of "safe" or non-flagged proteins.
- `known_amr.fasta` — sequences associated with antimicrobial resistance (AMR).
- `known_vf.fasta` — sequences associated with virulence factors (VF).

These files are used during training to construct a joint baseline distribution for the VAE.

### 3.2 Sequence validation

The helper script `validate_sequences.py` demonstrates how to check nucleotide sequences for illegal characters. It assumes that input files are line-based FASTA (one header line followed by one sequence line) and marks any sequence containing characters outside the canonical set `{A, T, C, G}`.

For a more robust pipeline, you would:

- Use Biopython (`SeqIO`) to parse FASTA rather than raw line indexing.
- Extend the validator to handle ambiguous nucleotides (e.g. N, R, Y) depending on your policy.
- Integrate validation into the main preprocessing or screening entrypoint so that invalid inputs are caught early.

### 3.3 Feature extraction

Feature extraction is implemented in `src/data_processor.py` via the `GenomicDataProcessor` class.

Key design choices:

- **Amino acid vocabulary.**
  - Uses a 20-letter amino acid alphabet: `ACDEFGHIKLMNPQRSTVWY`.
- **k-mer statistics.**
  - For each sequence and each k in `ks = [3, 4]`, all overlapping k-mers are enumerated.
  - For each k-mer present in the vocabulary product space, its frequency (count normalized by number of k-mers in the sequence) is stored in a fixed-length vector.
- **Physicochemical descriptors (ProtParam).**
  - Uses Biopython `ProteinAnalysis` to compute 9 global descriptors for each sequence:
    - net charge at pH 7.0
    - isoelectric point
    - aromaticity
    - instability index
    - GRAVY (hydropathy) score
    - molecular weight (scaled by 1/1000)
    - helix, turn, sheet secondary structure fractions
  - Any `X` characters are mapped to `A` to avoid parser errors.
- **Normalization.**
  - The final feature vector is L2-normalized before being returned.

The resulting feature dimension (e.g. 8437 for the current configuration) is used to parameterize the VAE input layer.

---

## 4. Model Architecture

The model is defined in `src/vae_model.py` as `GenomicVAE`, a fully connected variational autoencoder:

- **Encoder.**
  - Feed-forward stack with hidden sizes `[2048, 1024, 512, 256]`.
  - Each layer consists of `Linear → ReLU → Dropout(0.15)`.
  - Two separate linear heads output mean (`mu`) and log-variance (`logvar`) of the latent Gaussian.
- **Latent space.**
  - Latent dimension: 96 (configurable via constructor argument).
- **Decoder.**
  - Symmetric stack mirroring the encoder, again with `Linear → ReLU → Dropout(0.15)` blocks.
  - Final linear layer projects back to the original feature dimension.
  - Output is passed through a sigmoid squashing nonlinearity.

### 4.1 Training objective

The loss used during training (see `train.py`) combines:

- **Reconstruction loss.** Sum of squared errors between input `x` and reconstruction `recon`.
- **KL divergence term.** Weighted Kullback–Leibler divergence between the learned latent distribution and a unit Gaussian prior.

The overall objective is:

\[ \mathcal{L}(x) = \sum (x - \hat{x})^2 + \beta \cdot \mathrm{KL}(q(z|x) || \mathcal{N}(0, I)), \]

where \( \beta \) is set to 4.0 in this implementation to put more weight on a well-structured latent space.

---

## 5. Training Pipeline

### 5.1 Environment and dependencies

Dependencies are listed in `requirements.txt` and target Apple Silicon (M1/M2/M3/M4) with native Metal Performance Shaders (MPS) acceleration:

- `torch==2.4.1`
- `biopython>=1.83`
- `numpy>=1.24`
- `tqdm>=4.66`

A typical setup command would be:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 5.2 Training procedure

Training logic lives in `train.py`:

1. **Data loading.**
   - `load_seqs(path)` uses Biopython `SeqIO` to read sequences from FASTA.
   - Sequences shorter than 50 residues are discarded.
2. **Dataset construction.**
   - All sequences from `safe_baseline.fasta`, `known_amr.fasta`, and `known_vf.fasta` are concatenated.
   - A random subset is taken to limit memory and runtime (configurable cap, currently 10 sequences for debugging/demo purposes).
3. **Feature extraction.**
   - `GenomicDataProcessor.extract_features` converts sequences to fixed-length feature vectors.
4. **Model instantiation.**
   - `GenomicVAE(input_dim)` is created with input dimension inferred from the feature matrix.
   - Model is moved to `mps` device if available, otherwise CPU.
5. **Optimization.**
   - Optimizer: `AdamW` with learning rate `5e-4`.
   - Training loop runs for 80 epochs with mini-batches (batch size 256).
   - At each step, reconstruction and KL losses are combined; gradients are backpropagated and parameters updated.
6. **Monitoring.**
   - Every 10 epochs, the script prints mean loss per sequence.
   - Optional system stats (CPU load and temperature where available) are printed via `psutil` for basic resource monitoring.
7. **Checkpointing.**
   - After training, the model state dict is saved to `models/vae_best.pth`.

### 5.3 How to run training

```bash
python train.py --train
```

This will:

- Load the FASTA datasets under `data/`.
- Train the VAE model.
- Save weights to `models/vae_best.pth`.

---

---

## 6. Web Interface (Live Server)

To run the interactive web interface:

```bash
streamlit run app.py
```

This will launch a local server and open the application in your browser.

---

## 7. Screening / Inference Pipeline

The screening logic for new sequences is implemented in the `--screen` branch of `train.py`.

### 6.1 Workflow

1. **Load trained model.**
   - Instantiate `GenomicVAE` with the expected input dimension.
   - Load weights from `models/vae_best.pth`.
2. **Load and preprocess new sequences.**
   - Use `load_seqs` to read sequences from a user-specified FASTA file.
   - Apply the same `GenomicDataProcessor.extract_features` logic.
3. **Compute reconstruction errors.**
   - Disable gradients (`torch.no_grad()`).
   - Compute reconstructions for each feature vector.
   - For each sequence, compute mean squared reconstruction error across features.
4. **Thresholding and reporting.**
   - Compute a simple anomaly threshold: `mean(errors) + 3 * std(errors)`.
   - Sequences with error above the threshold are labeled as `THREAT`, others as `safe`.
   - A text report is printed, listing index, error, and status for each sequence, and the total number of sequences above threshold.

### 6.2 How to screen sequences

After training a model and saving it under `models/vae_best.pth`, run:

```bash
python train.py --screen path/to/new_sequences.fasta
```

This will print a **novelty detection report** to standard output.

For convenience, `screen.py` can be used as a thin wrapper or entrypoint to this functionality (e.g., importing and calling a `screen_fasta` function). In its simplest form, it can just forward arguments to `train.py --screen`.

---

## 7. Helper Scripts

### 7.1 `validate_sequences.py`

- Demonstrates validation of nucleotide FASTA files by rejecting any sequences containing characters outside `{A, T, C, G}`.
- Intended as an example of pre-screening input data to avoid malformed or low-quality sequences.

**Suggested improvements.**

- Replace manual line stepping with Biopython FASTA parsing (`SeqIO`).
- Make the script callable as a function and optionally expose a command-line interface taking `--input` and `--allow-ambiguous` flags.
- Integrate this validator into the training/screening pipeline so that invalid sequences are not silently used.

### 7.2 `format_fasta.py`

- Converts a structured text file (e.g. UniProt-style flat file) into FASTA format by scanning for lines starting with `ID` and concatenating subsequent sequence lines.
- Saves a new FASTA file under the specified output path.

**Suggested improvements.**

- Remove hard-coded file paths and accept `--input` and `--output` paths from the command line.
- Add basic integrity checks (e.g. warn if no sequences are found, or if sequences contain non-standard characters).
- Optionally log how many entries were successfully converted.

---

## 8. Biosafety Considerations and Limitations

From a biosafety perspective, the following points should be highlighted explicitly in any report or documentation:

- **No guarantee of safety.** A `safe` label here means only "not strongly out-of-distribution relative to the training baseline"; it does *not* mean biologically harmless.
- **Model blind spots.**
  - The model may fail to flag sequences that are harmful but lie close to the training distribution.
  - Cleverly engineered sequences might evade such simple anomaly detectors.
- **Data provenance.**
  - Clearly document sources and curation steps for the `safe_baseline`, `known_amr`, and `known_vf` datasets.
  - Consider whether including known harmful sequences in the baseline may make the model less sensitive to those classes.
- **Operational integration.**
  - If used at all in real pipelines, this model should be one component alongside curated blacklists, motif search, and expert review.
  - False positives and false negatives should be monitored and fed back into model revisions.

---

## 9. Suggested Improvements to Code Structure

To make the codebase more presentable and robust for a biosafety project, consider the following refactors:

1. **Create a clear package structure.**
   - Keep core logic under `src/` (data processing, models, utils).
   - Restrict top-level scripts (`train.py`, `screen.py`, `validate_sequences.py`, `format_fasta.py`) to thin command-line wrappers that call into library functions.

2. **Turn scripts into CLIs.**
   - Use `argparse` consistently:
     - `train.py`: arguments such as `--epochs`, `--batch-size`, `--beta-kl`, `--subset-size`, `--device`.
     - `screen.py`: arguments such as `--input-fasta`, `--model-path`, `--threshold-mode` (`mean+3std` or user-defined).
     - `validate_sequences.py` and `format_fasta.py`: `--input`, `--output`, optional flags.

3. **Centralize configuration.**
   - Add a `config.py` or YAML file storing model hyperparameters, data paths, and thresholds so they are not duplicated across scripts.

4. **Improve logging.**
   - Replace ad-hoc `print` statements with Python's `logging` module.
   - Log key metadata: timestamp, Git commit (if available), dataset sizes, hyperparameters, device, and final performance metrics.

5. **Add tests.**
   - Use `pytest` to add simple tests for:
     - `GenomicDataProcessor.extract_features` (dimensionality, handling of short sequences, no NaNs).
     - `GenomicVAE` forward pass (shape consistency, no runtime errors on CPU).

6. **Document entrypoints.**
   - Expand the project `README` to describe:
     - Installation.
     - How to train a model.
     - How to run screening.
     - Limitations and biosafety disclaimers.

---

## 10. How to Generate a PDF Report

This file (`REPORT.md`) is designed so it can be converted directly into a PDF for inclusion in a project submission or appendix.

### 10.1 Using Pandoc (recommended)

1. **Install Pandoc** (if not already installed). On macOS with Homebrew:

   ```bash
   brew install pandoc
   ```

2. **Generate a PDF** from the project root:

   ```bash
   pandoc REPORT.md -o REPORT.pdf
   ```

   Optionally, you can add a title page and better typography via a template, e.g.:

   ```bash
   pandoc REPORT.md -o REPORT.pdf \
     --from markdown \
     --pdf-engine=xelatex \
     -V mainfont="Times New Roman" \
     -V geometry:margin=1in
   ```

### 10.2 Using a Markdown-capable editor

Many editors (VS Code extensions, Typora, etc.) can open `REPORT.md` and export to PDF via their GUI. Open the file, choose **Export → PDF**, and save the output.

---

## 11. Summary

- Implemented a VAE-based anomaly detector over protein sequence features for biosafety-oriented novelty screening.
- Provided a transparent feature engineering pipeline combining k-mer statistics and physicochemical descriptors.
- Documented training and inference workflows, along with their limitations and appropriate biosafety framing.
- Suggested concrete improvements (CLI refactors, configuration management, validation, and logging) to make the codebase more robust and presentable for a biosafety-focused project.
