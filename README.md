# Awesome fMRI Decoding (Categorized)

A curated list of **fMRI-based brain decoding** resources, focusing on **reconstructing images, videos, language, and audio from brain activity**.

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

- [0. Survey & Background](#0-survey--background)
  - [0.1 Survey & Overview](#01-survey--overview)
  - [0.2 Tutorials & Intro Reading](#02-tutorials--intro-reading)
- [1. Resources](#1-resources)
  - [1.1 Datasets & Benchmarks](#11-datasets--benchmarks)
  - [1.2 Toolboxes & Codebases](#12-toolboxes--codebases)
  - [1.3 Other Awesome Lists](#13-other-awesome-lists)
- [2. fMRI Decoding Methods (by Task)](#2-fmri-decoding-methods-by-task)
  - [2.1 Brain → Image (Static Vision)](#21-brain--image-static-vision)
    - [2.1.1 Early / Pre-deep](#211-early--pre-deep)
    - [2.1.2 GAN / VAE–based](#212-gan--vae-based)
    - [2.1.3 Diffusion-based](#213-diffusion-based)
    - [2.1.4 Cross-subject / Few-shot / MoE](#214-cross-subject--few-shot--moe)
  - [2.2 Brain → Video / Dynamic Scene](#22-brain--video--dynamic-scene)
  - [2.3 Brain → Text / Narrative](#23-brain--text--narrative)
  - [2.4 Brain → Audio / Music](#24-brain--audio--music)
  - [2.5 Multimodal & Foundation-Model-based Decoding](#25-multimodal--foundation-model-based-decoding)
  - [2.6 Clinical / Cognitive / Mental-State Decoding](#26-clinical--cognitive--mental-state-decoding)
- [3. Related fMRI Modeling](#3-related-fmri-modeling)
  - [3.1 Visual → fMRI Encoding & Data Augmentation](#31-visual--fmri-encoding--data-augmentation)
  - [3.2 Multimodal fMRI + EEG / MEG](#32-multimodal-fmri--eeg--meg)
  - [3.3 Representation Alignment & Analysis](#33-representation-alignment--analysis)
- [Contributing](#contributing)

---

## 0. Survey & Background

### 0.1 Survey & Overview

**A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli**  
[📄 Paper](https://arxiv.org/abs/2503.15978) • [💻 Project](https://github.com/LpyNow/BrainDecodingImage) • _[Survey]_

**Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy**  
[📄 Paper](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO) • _[Survey]_

**Deep Generative Models in Brain Encoding and Decoding**  
[📄 Paper](https://doi.org/10.1016/j.eng.2019.03.011) • _[Survey]_

### 0.2 Tutorials & Intro Reading

*(Feel free to add blog posts, lecture notes, or tutorial-style papers here.)*

---

## 1. Resources

### 1.1 Datasets & Benchmarks

**Natural Scenes Dataset (NSD)** – Large-scale 7T visual fMRI dataset.  
[🌐 Website](https://naturalscenesdataset.org/) • [📂 Data](https://osf.io/9pjky/) • _[Vision]_

**Deep Image Reconstruction (DIR) dataset** – Data for Kamitani *Deep Image Reconstruction* experiments.  
[📂 Data](https://openneuro.org/datasets/ds001506) • _[Vision]_

**Narratives / Story listening datasets** – Long-form spoken stories for language / narrative decoding.  
[🌐 Website](https://www.narrativeslab.org/) • [📂 Data](https://openneuro.org/datasets/ds002345) • _[Audio] [Language]_

**Semantic reconstruction of continuous language – dataset** – Accompanies Tang et al. fMRI-to-text work.  
[📂 Data](https://openneuro.org/datasets/ds003020) • _[Audio] [Language]_

*(More welcome: Vim-1, BOLD5000, GOD, movie-watching datasets, etc.)*

### 1.2 Toolboxes & Codebases

**DeepImageReconstruction** – End-to-end pipeline for visual fMRI → image reconstruction.  
[💻 Code](https://github.com/KamitaniLab/DeepImageReconstruction)

**semantic-decoding** – Implementation of semantic reconstruction of continuous language from fMRI.  
[💻 Code](https://github.com/HuthLab/semantic-decoding)

**MindReader** – CLIP + StyleGAN2–based fMRI visual reconstructor.  
[💻 Code](https://github.com/yuvalsim/MindReader)

**MindEye2** – Shared-subject fMRI-to-image reconstruction with 1 hour of data per new subject.  
[💻 Code](https://github.com/MedARC-AI/MindEyeV2)

*(You can also list fMRIPrep, nilearn, etc. here if you want preprocessing tools.)*

### 1.3 Other Awesome Lists

**awesome-brain-decoding** – General brain decoding list (EEG / MEG / fMRI / ECoG).  
[📦 GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)

**Awesome Brain Encoding & Decoding** – Mixed encoding/decoding collection.  
[📦 GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

**Awesome Brain Graph Learning with GNNs** – GNN-based brain graph learning and analysis.  
[📦 GitHub](https://github.com/XuexiongLuoMQ/Awesome-Brain-Graph-Learning-with-GNNs)

---

## 2. fMRI Decoding Methods (by Task)

> For each paper, tags roughly indicate task & method, e.g. _[Brain→Image] [Diffusion] [NSD]_.

---

### 2.1 Brain → Image (Static Vision)

#### 2.1.1 Early / Pre-deep

**Reconstructing Natural Scenes from fMRI Patterns Using Hierarchical Visual Features**  
[📄 Paper](https://doi.org/10.1016/j.neuroimage.2010.07.063) • _[Brain→Image] [Early]_

**Visual Experience Reconstruction from Movie fMRI**  
[📄 Paper](https://doi.org/10.1016/j.cub.2011.01.031) • _[Brain→Video] [Early]_

#### 2.1.2 GAN / VAE–based

**Deep Image Reconstruction from Human Brain Activity**  
[📄 Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) • [💻 Code](https://github.com/KamitaniLab/DeepImageReconstruction) • [📂 Dataset](https://openneuro.org/datasets/ds001506) • _[Brain→Image] [GAN/VAE]_

**From Voxels to Pixels and Back: Self-supervision in Natural-Image Reconstruction from fMRI**  
[📄 Paper](https://arxiv.org/abs/1907.02431) • [💻 Code](https://github.com/WeizmannVision/ssfmri2im) • _[Brain→Image] [Self-supervised]_

**Reconstructing Natural Scenes from fMRI Patterns Using BigBiGAN**  
[📄 Paper](https://arxiv.org/abs/2011.12243) • _[Brain→Image] [BigGAN]_

#### 2.1.3 Diffusion-based

**Brain-Diffuser: Natural Scene Reconstruction from fMRI Signals Using Generative Latent Diffusion**  
[📄 Paper](https://www.nature.com/articles/s41598-023-42891-8) • [💻 Code](https://github.com/ozcelikfu/brain-diffuser) • [📂 Dataset: NSD](https://naturalscenesdataset.org/) • _[Brain→Image] [Diffusion] [NSD]_

**Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)**  
[📄 Paper](https://arxiv.org/abs/2305.18274) • [🌐 Project](https://medarc-ai.github.io/mindeye/) • [💻 Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) • _[Brain→Image] [Diffusion] [Contrastive]_

**MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion**  
[📄 Paper](https://arxiv.org/abs/2308.04249) • [💻 Code](https://github.com/YingxingLu/MindDiffuser) • _[Brain→Image] [Diffusion]_

**NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction**  
[📄 Paper](https://arxiv.org/abs/2401.01713) • [💻 Code](https://github.com/neu-diffusion/NeuralDiffuser) • _[Brain→Image] [Diffusion]_

**Mental Image Reconstruction from Human Brain Activity**  
[📄 Paper](https://www.sciencedirect.com/science/article/pii/S0893608023006470) • _[Brain→Image] [Diffusion]_

**MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data**  
[📄 Paper](https://arxiv.org/abs/2403.11207) • [🌐 Project](https://medarc-ai.github.io/mindeye2/) • [💻 Code](https://github.com/MedARC-AI/MindEyeV2) • _[Brain→Image] [Diffusion] [Shared-subject]_

**Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks**  
[📄 Paper](https://openreview.net/forum?id=K6ijewH34E) • [📄 PDF](https://openreview.net/pdf/167d5c3c08cdd7367883eeec0b26002c059215f8.pdf) • _[Brain→Image] [Diffusion] [Semantic Bottleneck] [NeurIPS 2025]_  

#### 2.1.4 Cross-subject / Few-shot / MoE

**ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding**  
[📄 Paper](https://arxiv.org/abs/2510.27128) • [📄 PDF](https://openreview.net/pdf/7a4f583ef54685490be5c58986a3ad803aac087c.pdf) • [💻 Code](https://github.com/xmed-lab/ZEBRA) • _[Brain→Image] [Diffusion] [Cross-Subject] [NeurIPS 2025]_

**MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding**  
[📄 Paper](https://arxiv.org/abs/2505.15946) • [🌐 OpenReview](https://openreview.net/forum?id=fYSPRGmS6l) • [💻 Code](https://github.com/yuxiangwei0808/MoRE-Brain) • _[Brain→Image] [MoE] [Cross-Subject] [NeurIPS 2025]_

*(Other cross-subject / few-shot works can also be added here.)*

---

### 2.2 Brain → Video / Dynamic Scene

**Visual Experience Reconstruction from Movie fMRI**  
[📄 Paper](https://doi.org/10.1016/j.cub.2011.01.031) • _[Brain→Video] [Early]_

**CLSR: Decoding Complex Video and Story Stimuli from fMRI**  
[📄 Paper](https://doi.org/10.1038/s41593-023-01327-2) • _[Brain→Video] [Brain→Text]_

*(Add additional movie / video reconstruction and video-captioning decoders here.)*

---

### 2.3 Brain → Text / Narrative

**Semantic Reconstruction of Continuous Language from Non-Invasive Brain Recordings**  
[📄 Paper](https://www.nature.com/articles/s41593-023-01304-9) • [💻 Code](https://github.com/HuthLab/semantic-decoding) • [📂 Dataset](https://openneuro.org/datasets/ds003020) • _[Brain→Text] [Narrative]_

**UniCoRN: Unified Cognitive Signal ReconstructioN Bridging Cognitive Signals and Human Language**  
[📄 Paper](https://arxiv.org/abs/2307.05355) • _[Brain→Text] [EEG+fMRI] [LLM]_

**Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)**  
[📄 Paper](https://openreview.net/forum?id=REIo9ZLSYo) • [📄 PDF](https://openreview.net/pdf?id=REIo9ZLSYo) • [💻 Code](https://github.com/WENXUYUN/CogReader) • _[Brain→Text] [LLM] [NeurIPS 2025 Spotlight]_

*(More language / narrative decoding works welcome.)*

---

### 2.4 Brain → Audio / Music

*(Reserved for fMRI decoding of auditory scenes, speech, and music. Add works on music genre/affect, sound-category decoding, etc.)*

---

### 2.5 Multimodal & Foundation-Model-based Decoding

**MindReader: Reconstructing Complex Images from Brain Activities**  
[📄 Paper](https://arxiv.org/abs/2209.12951) • [💻 Code](https://github.com/yuvalsim/MindReader) • _[Brain→Image] [CLIP] [StyleGAN2]_

**UMBRAE: Unified Multimodal Brain Decoding**  
[📄 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf) • [🌐 Project](https://weihaox.github.io/UMBRAE) • _[Brain→Image] [Brain→Text] [Multimodal]_

*(You can also put brain-conditional multimodal AIGC, VLM/LLM-based unified decoders here.)*

---

### 2.6 Clinical / Cognitive / Mental-State Decoding

*(Reserved for works decoding emotion, cognitive load, disease markers, etc., from fMRI. Add when you curate them.)*

---

## 3. Related fMRI Modeling

These works are closely related to decoding but not always “direct Brain→stimulus” decoders.

### 3.1 Visual → fMRI Encoding & Data Augmentation

**SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning**  
[📄 Paper](https://arxiv.org/abs/2508.10298) • [🌐 OpenReview](https://openreview.net/forum?id=ZTHYaSxqmq) • _[Visual→fMRI] [Encoding] [Probabilistic] [NeurIPS 2025 Poster]_

*(Add other visual→fMRI encoders and synthetic-fMRI data augmentation works here.)*

### 3.2 Multimodal fMRI + EEG / MEG

**Joint Modeling of fMRI and EEG Imaging Using Ordinary Differential Equation-Based Hypergraph Neural Networks (FE-NET)**  
[📄 PDF](https://openreview.net/pdf/053f8c5a43f7051852d82cdcb8ab742f69065ea2.pdf) • _[fMRI+EEG] [Hypergraph] [Neural ODE] [NeurIPS 2025]_  

*(Add more multimodal modeling methods, e.g., fMRI+DTI+sMRI analyses, jICA, etc.)*

### 3.3 Representation Alignment & Analysis

*(For encoding-only LM-alignment, RSA / brain-score analysis, representational comparisons between fMRI and deep networks. To be filled.)*

---

## Contributing

Contributions are welcome!  

If you want to add or update a paper:

1. Make sure it is **fMRI-related** and either:
   - a decoding method (preferably Brain→Image/Text/Video/Audio), or  
   - a strongly related modeling work (encoding / data augmentation / multimodal modeling).
2. Choose the appropriate section (and sub-section).
3. Follow this format:

   ```markdown
   **Paper Title**  
   [📄 Paper](...) • [💻 Code](...) • [📂 Dataset](...) • _[Brain→Image] [Diffusion] [NSD]_
