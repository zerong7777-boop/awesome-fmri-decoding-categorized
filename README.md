# Awesome fMRI Decoding (Categorized)

A curated list of **fMRI-based brain decoding** papers, datasets and code, focusing on **reconstructing images / videos / language / audio from brain activity**.

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

1. [Survey & Overview](#survey--overview)
2. [Datasets & Benchmarks](#datasets--benchmarks)
3. [Language & Narrative Decoding (Brain → Text)](#language--narrative-decoding-brain--text)
4. [Visual Image Reconstruction (Brain → Image)](#visual-image-reconstruction-brain--image)
   - [Early & Pre-generative Works](#early--pre-generative-works)
   - [GAN / VAE–based](#gan--vae-based)
   - [Diffusion-based](#diffusion-based)
   - [Cross-Subject & Mixture-of-Experts](#cross-subject--mixture-of-experts)
5. [Video & Dynamic Scene Decoding](#video--dynamic-scene-decoding)
6. [Visual-to-fMRI Encoding & Data Augmentation](#visual-to-fmri-encoding--data-augmentation)
7. [Audio & Music Decoding](#audio--music-decoding)
8. [Multimodal & Foundation-Model-based Decoding](#multimodal--foundation-model-based-decoding)
9. [Clinical / Cognitive & Mental-State Decoding](#clinical--cognitive--mental-state-decoding)
10. [Toolboxes, Tutorials & Awesome Lists](#toolboxes-tutorials--awesome-lists)
11. [Contributing](#contributing)
12. [License](#license)

---

## Survey & Overview

**A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli** (arXiv 2025) – Recent survey of fMRI-based decoding for images, text, audio and video, with taxonomy of tasks and models.  [📄 Paper](https://arxiv.org/abs/2503.15978)  •  [💻 Project Code](https://github.com/LpyNow/BrainDecodingImage)

**Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy** (IEEE TAI 2025) – Survey of brain-conditional generative models (image, audio, text) with taxonomy and evaluation protocols.  [📄 Paper](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)

**Deep Generative Models in Brain Encoding and Decoding** (Engineering 2019) – Early review of using deep generative models for brain encoding/decoding.  [📄 Paper](https://doi.org/10.1016/j.eng.2019.03.011)

---

## Datasets & Benchmarks

**Natural Scenes Dataset (NSD)** – High-resolution 7T fMRI while subjects view thousands of natural images; widely used for visual reconstruction.  [🌐 Website](https://naturalscenesdataset.org/)  •  [📂 Data](https://osf.io/9pjky/)

**Deep Image Reconstruction (DIR) dataset** – fMRI data for the Kamitani *Deep Image Reconstruction* study.  [📂 Data](https://openneuro.org/datasets/ds001506)

**Narratives / Story Listening datasets** – Long-form spoken story comprehension, used for fMRI-to-text decoding.  [🌐 Narratives Lab](https://www.narrativeslab.org/)  •  [📂 Data](https://openneuro.org/datasets/ds002345)

**Semantic reconstruction of continuous language – dataset** – Data accompanying Tang et al., Nat Neurosci 2023.  [📂 Data](https://openneuro.org/datasets/ds003020)

(*Feel free to add more datasets: Vim-1, BOLD5000, GOD, BOLD500K, CelebrityFace, movie-watching datasets, etc.*)

---

## Language & Narrative Decoding (Brain → Text)

**Semantic Reconstruction of Continuous Language from Non-invasive Brain Recordings** (Nat Neurosci 2023) – fMRI-to-text decoder that reconstructs continuous language from cortical semantic representations.  [📄 Paper](https://www.nature.com/articles/s41593-023-01304-9)  •  [💻 Code](https://github.com/HuthLab/semantic-decoding)  •  [📂 Dataset](https://openneuro.org/datasets/ds003020)

**UniCoRN: Unified Cognitive Signal ReconstructioN Bridging Cognitive Signals and Human Language** (ACL 2023) – Proposes the fMRI2text task and a unified encoder for fMRI/EEG with a pretrained language model decoder.  [📄 Paper](https://arxiv.org/abs/2307.05355)  •  [💻 Code](https://github.com/DUTIR-ESLab/UniCoRN)

**Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)** (NeurIPS 2025 Spotlight) – NeurIPS 2025 work that mimics human segmented language comprehension; decodes long narratives with segment-wise incremental decoding and wrap-up integration.  [📄 Paper](https://openreview.net/forum?id=REIo9ZLSYo)  •  [💻 Code](https://github.com/WENXUYUN/CogReader)

(*You can add more narrative / language decoding works here, e.g., character-level decoders, other fMRI-to-text or EEG-to-text models.*)

---

## Visual Image Reconstruction (Brain → Image)

### Early & Pre-generative Works

**Reconstructing Natural Scenes from fMRI Patterns Using Hierarchical Visual Features** (NeuroImage 2011) – Uses Gabor and Gist-like features plus Bayesian decoding to reconstruct natural images from early visual cortex.  [📄 Paper](https://doi.org/10.1016/j.neuroimage.2010.07.063)

**Visual Experience Reconstruction from Movie fMRI** (Current Biology 2011) – Classic work reconstructing low-resolution movies from fMRI responses during movie watching.  [📄 Paper](https://doi.org/10.1016/j.cub.2011.01.031)

### GAN / VAE–based

**Deep Image Reconstruction from Human Brain Activity** (PLoS Comput Biol 2019) – Maps fMRI to DNN feature space and reconstructs images via a deep generative model.  [📄 Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)  •  [💻 Code](https://github.com/KamitaniLab/DeepImageReconstruction)  •  [📂 Dataset](https://openneuro.org/datasets/ds001506)

**From Voxels to Pixels and Back: Self-supervision in Natural-Image Reconstruction from fMRI** (NeurIPS 2019) – Self-supervised framework aligning fMRI with deep visual features for image reconstruction.  [📄 Paper](https://arxiv.org/abs/1907.02431)  •  [💻 Code](https://github.com/WeizmannVision/ssfmri2im)

**Reconstructing Natural Scenes from fMRI Patterns Using BigBiGAN** (preprint) – Uses BigBiGAN latent space for fMRI-to-image reconstruction.  [📄 Paper](https://arxiv.org/abs/2011.12243)

### Diffusion-based

**Brain-Diffuser: Natural Scene Reconstruction from fMRI Signals Using Generative Latent Diffusion** (Sci Rep 2023) – Conditions latent diffusion on fMRI-aligned deep features to reconstruct images on NSD.  [📄 Paper](https://www.nature.com/articles/s41598-023-42891-8)  •  [💻 Code](https://github.com/ozcelikfu/brain-diffuser)  •  [📂 Dataset: NSD](https://naturalscenesdataset.org/)

**Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)** (NeurIPS 2023) – Contrastive fMRI-to-CLIP encoder plus diffusion prior for high-fidelity reconstructions.  [📄 Paper](https://arxiv.org/abs/2305.18274)  •  [🌐 Project](https://medarc-ai.github.io/mindeye/)  •  [💻 Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)

**MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion** (ACM MM 2023) – Two-stage semantic and structural diffusion with extra controllability.  [📄 Paper](https://arxiv.org/abs/2308.04249)  •  [💻 Code](https://github.com/YingxingLu/MindDiffuser)

**NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction** (IEEE TIP 2025) – Introduces neuroscience-inspired guidance signals to steer diffusion sampling using fMRI activity.  [📄 Paper](https://arxiv.org/abs/2401.01713)  •  [💻 Code](https://github.com/neu-diffusion/NeuralDiffuser)

**Mental Image Reconstruction from Human Brain Activity** (Neural Networks 2024) – Uses diffusion priors with additional perceptual and semantic constraints for mental image reconstruction.  [📄 Paper](https://www.sciencedirect.com/science/article/pii/S0893608023006470)

**Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks** (NeurIPS 2025 Poster) – Inserts an explicit semantic bottleneck into the fMRI-to-image pipeline for concept-level interpretability, building on diffusion-based visual decoders.  [📄 Paper](https://openreview.net/forum?id=K6ijewH34E)  •  [📄 PDF](https://openreview.net/pdf/167d5c3c08cdd7367883eeec0b26002c059215f8.pdf)  •  [🌐 NeurIPS page](https://neurips.cc/virtual/2025/poster/118670)

### Cross-Subject & Mixture-of-Experts

**ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding** (NeurIPS 2025 Poster) – Disentangles subject-specific and semantic components of fMRI via adversarial training, enabling zero-shot cross-subject image reconstruction.  [📄 Paper](https://arxiv.org/abs/2510.27128)  •  [📄 PDF](https://openreview.net/pdf/7a4f583ef54685490be5c58986a3ad803aac087c.pdf)  •  [💻 Code](https://github.com/xmed-lab/ZEBRA)

**MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding** (NeurIPS 2025 Poster) – Routed mixture-of-experts mapping fMRI to CLIP space with subject-specific routers and diffusion decoder; emphasizes interpretability of expert routing.  [📄 Paper](https://arxiv.org/abs/2505.15946)  •  [🌐 OpenReview](https://openreview.net/forum?id=fYSPRGmS6l)  •  [💻 Code](https://github.com/yuxiangwei0808/MoRE-Brain)

(*Feel free to add other recent cross-subject / few-shot / foundation-model-based decoders here, such as MindEye2, BrainSD, MinD-Vis, etc.*)

---

## Video & Dynamic Scene Decoding

**Visual Experience Reconstruction from Movie fMRI** (Current Biology 2011) – Reconstructs natural movies from early visual cortex responses.  [📄 Paper](https://doi.org/10.1016/j.cub.2011.01.031)

**CLSR: Decoding Complex Video and Story Stimuli from fMRI** (Nat Neurosci 2023) – Large-scale movie & story dataset with joint video / text decoding from fMRI.  [📄 Paper](https://doi.org/10.1038/s41593-023-01327-2)

(*Add here movie-watching, dynamic scene and video-to-text fMRI decoders.*)

---

## Visual-to-fMRI Encoding & Data Augmentation

> These works learn **image → fMRI** mappings (encoding) and often use them to synthesize fMRI for data augmentation or analysis, which can improve fMRI-to-image decoders.

**SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning** (NeurIPS 2025 Poster / arXiv 2025) – Probabilistic visual-to-fMRI encoder with BrainVAE and semantic-to-neural mapper; synthesized fMRI improves downstream decoding.  [📄 Paper](https://arxiv.org/abs/2508.10298)  •  [📄 PDF](https://openreview.net/pdf/3971b93a4f08a3549d29904c63d514e0df961001.pdf)

(*You can add other encoding / data-augmentation works here, e.g., fMRI encoders used to generate synthetic training data for decoders.*)

---

## Audio & Music Decoding

(*Reserved for fMRI-based decoding of auditory scenes, speech, and music – e.g., music genre / affect decoding, sound category decoding, auditory cortex modeling.  Contributions welcome.*)

---

## Multimodal & Foundation-Model-based Decoding

**MindReader: Reconstructing Complex Images from Brain Activities** (NeurIPS 2022) – Uses CLIP space and StyleGAN2 as generative prior for complex image reconstruction from fMRI.  [📄 Paper](https://arxiv.org/abs/2209.12951)  •  [💻 Code](https://github.com/yuvalsim/MindReader)

**UMBRAE: Unified Multimodal Brain Decoding** (ECCV 2024) – Unified framework that decodes images and text from brain activity using multimodal foundation models.  [📄 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf)  •  [🌐 Project](https://weihaox.github.io/UMBRAE)

**Joint Modeling of fMRI and EEG Imaging Using Ordinary Differential Equation-based Hypergraph Neural Networks (FE-NET)** (NeurIPS 2025) – Jointly models asynchronous fMRI-EEG data via GAN-based hypergraph generation and Neural ODE–based temporal dynamics for multimodal brain decoding.  [📄 Paper](https://openreview.net/pdf/053f8c5a43f7051852d82cdcb8ab742f69065ea2.pdf)

(*You can also list multimodal models combining fMRI with EEG, MEG, sMRI, DTI, text or audio, as well as large foundation models for brain decoding / encoding.*)

---

## Clinical / Cognitive & Mental-State Decoding

(*Reserved for works that apply fMRI decoding to clinical or cognitive states – e.g., decoding emotion, disease markers, psychiatric conditions, or cognitive load.  Add your favorite papers here.*)

---

## Toolboxes, Tutorials & Awesome Lists

**DeepImageReconstruction** – Official code for Kamitani *Deep Image Reconstruction*, including full pipeline from preprocessing to reconstruction.  [💻 Code](https://github.com/KamitaniLab/DeepImageReconstruction)

**semantic-decoding** – Official HuthLab code for semantic reconstruction of continuous language from fMRI.  [💻 Code](https://github.com/HuthLab/semantic-decoding)

**MindReader (code)** – Implementation of MindReader CLIP-based decoder with StyleGAN2 generator.  [💻 Code](https://github.com/yuvalsim/MindReader)

**awesome-brain-decoding** – A broader awesome list covering EEG / fMRI / ECoG decoding.  [📦 GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)

**Awesome Brain Encoding & Decoding** – Another general collection of brain encoding / decoding papers.  [📦 GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

(*You can also list preprocessing toolboxes, fMRI loaders, visualization tools, and educational tutorials here, such as fMRIPrep, nilearn, AFNI / SPM toolchains, etc.*)

---

## Contributing

Contributions are very welcome!  Before opening a PR:

1. Check that the work is **about fMRI-based brain decoding or closely related encoding / data-augmentation** (or has a strong fMRI component).
2. Choose the most appropriate category and sub-category.
3. Use the following format for each entry:

   ```markdown
   **Paper Title** (Conf./Journal Year) – Short one-sentence description.  [📄 Paper](...)  •  [💻 Code](...)  •  [📂 Dataset](...)
