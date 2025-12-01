# Awesome fMRI Decoding (Categorized)

A curated list of **fMRI-based brain decoding** papers and resources, focusing on **reconstructing images / videos / language / audio from brain activity** (plus a few closely related visual-encoding works that directly support decoding).

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

- [1. Surveys and Overviews](#1-surveys-and-overviews)
- [2. Datasets and Benchmarks](#2-datasets-and-benchmarks)
- [3. Language / Narrative Decoding (Brain → Text)](#3-language--narrative-decoding-brain--text)
- [4. Visual Image Reconstruction (Brain → Image)](#4-visual-image-reconstruction-brain--image)
  - [4.1 Classical and Pre-Generative](#41-classical-and-pre-generative)
  - [4.2 GAN / VAE-based](#42-gan--vae-based)
  - [4.3 Diffusion-based Reconstruction](#43-diffusion-based-reconstruction)
  - [4.4 Cross-Subject and Generalizable Decoding](#44-cross-subject-and-generalizable-decoding)
  - [4.5 Interpretability and Concept-Level Decoding](#45-interpretability-and-concept-level-decoding)
  - [4.6 Visual-to-fMRI Synthesis and Data Augmentation](#46-visual-to-fmri-synthesis-and-data-augmentation)
- [5. Video and Dynamic Scene Decoding](#5-video-and-dynamic-scene-decoding)
- [6. Multimodal and Foundation-Model-based Decoding](#6-multimodal-and-foundation-model-based-decoding)
- [7. Audio and Music Decoding](#7-audio-and-music-decoding)
- [8. Clinical / Cognitive and Mental-State Decoding](#8-clinical--cognitive-and-mental-state-decoding)
- [9. Toolboxes and Awesome Lists](#9-toolboxes-and-awesome-lists)
- [10. Contributing](#10-contributing)
- [11. License](#11-license)

---

## 1. Surveys and Overviews

> **Scope:** 全局综述 / review / tutorial，介绍 fMRI 解码或 brain-conditional 生成的大图景。

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[[arXiv 2025](https://arxiv.org/abs/2503.15978)]

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[[IEEE TAI 2025](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)] [[Project](https://github.com/MichaelMaiii/AIGC-Brain)]

Deep Generative Models in Brain Encoding and Decoding  
[[Engineering 2019](https://doi.org/10.1016/j.eng.2019.03.011)]

---

## 2. Datasets and Benchmarks

> **Scope:** 公开的 fMRI 数据集 / benchmark（视觉、语言、音频等），不按方法，只按数据。

Natural Scenes Dataset (NSD)  
[[Website](https://naturalscenesdataset.org/)] [[Data](https://osf.io/9pjky/)]

Deep Image Reconstruction (DIR) dataset  
[[OpenNeuro ds001506](https://openneuro.org/datasets/ds001506)]

Narratives / Story listening datasets  
[[Website](https://www.narrativeslab.org/)] [[OpenNeuro ds002345](https://openneuro.org/datasets/ds002345)]

Semantic reconstruction of continuous language – dataset  
[[OpenNeuro ds003020](https://openneuro.org/datasets/ds003020)]

*(More welcome: Vim-1, BOLD5000, GOD / THINGS, CelebrityFace, various movie fMRI datasets, etc.)*

---

## 3. Language / Narrative Decoding (Brain → Text)

> **Scope:** 输出是「文本」：句子、段落、故事摘要、caption 等（fMRI → text）。

Semantic reconstruction of continuous language from non-invasive brain recordings  
[[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9)] [[Code](https://github.com/HuthLab/semantic-decoding)] [[Dataset](https://openneuro.org/datasets/ds003020)]

Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)  
[[NeurIPS 2025 Spotlight](https://openreview.net/forum?id=REIo9ZLSYo)] [[PDF](https://openreview.net/pdf?id=REIo9ZLSYo)]

*(More fMRI→text / narrative decoding works can be added here.)*

---

## 4. Visual Image Reconstruction (Brain → Image)

> **Scope:** 输出是「静态图像」。根据方法特点再细分为 4.1–4.6。

### 4.1 Classical and Pre-Generative

> 早期方法：不用现代 deep generative models 的图像重建 / 解码。

Reconstructing Natural Scenes from fMRI Patterns using Hierarchical Visual Features  
[[NeuroImage 2011](https://doi.org/10.1016/j.neuroimage.2010.07.063)]

---

### 4.2 GAN / VAE-based

> 主生成器是 GAN / VAE / self-supervised convnets，而不是 diffusion。

Deep image reconstruction from human brain activity  
[[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)] [[Code](https://github.com/KamitaniLab/DeepImageReconstruction)] [[Dataset](https://openneuro.org/datasets/ds001506)]

From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI  
[[NeurIPS 2019](https://arxiv.org/abs/1907.02431)] [[Code](https://github.com/WeizmannVision/ssfmri2im)]

Reconstructing Natural Scenes from fMRI Patterns using BigBiGAN  
[[arXiv 2020](https://arxiv.org/abs/2011.12243)]

---

### 4.3 Diffusion-based Reconstruction

> 使用 diffusion / latent diffusion / Stable Diffusion 作为生成 prior 的 fMRI→image 方法。

High-resolution image reconstruction with latent diffusion models from human brain activity  
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html)] [[Project](https://sites.google.com/view/stablediffusion-with-brain/)] [[Code](https://github.com/yu-takagi/StableDiffusionReconstruction)]

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)  
[[NeurIPS 2023](https://arxiv.org/abs/2305.18274)] [[Project](https://medarc-ai.github.io/mindeye/)] [[Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)]

Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion  
[[Scientific Reports 2023](https://www.nature.com/articles/s41598-023-42891-8)] [[arXiv](https://arxiv.org/abs/2303.05334)] [[Code](https://github.com/ozcelikfu/brain-diffuser)]

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion  
[[ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3613819)] [[arXiv](https://arxiv.org/abs/2308.04249)] [[Code](https://github.com/YingxingLu/MindDiffuser)]

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction  
[[IEEE TIP 2025](https://ieeexplore.ieee.org/document/10749645)] [[arXiv](https://arxiv.org/abs/2401.01713)]

Mental image reconstruction from human brain activity  
[[Neural Networks 2024](https://www.sciencedirect.com/science/article/pii/S0893608023006470)]

---

### 4.4 Cross-Subject and Generalizable Decoding

> 依然是 Brain→Image，但重点在 **跨被试泛化 / 少样本适配 / MoE / shared-subject**。

MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data  
[[ICML 2024](https://proceedings.mlr.press/v235/scotti24a.html)] [[arXiv](https://arxiv.org/abs/2403.11207)] [[Project](https://medarc-ai.github.io/mindeye2/)] [[Code](https://github.com/MedARC-AI/MindEyeV2)]

ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding  
[[OpenReview](https://openreview.net/pdf/7a4f583ef54685490be5c58986a3ad803aac087c)] [[Code](https://github.com/xmed-lab/ZEBRA)]

MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding  
[[OpenReview](https://openreview.net/forum?id=fYSPRGmS6l)] [[arXiv](https://arxiv.org/abs/2505.15946)] [[Code](https://github.com/yuxiangwei0808/MoRE-Brain)]

---

### 4.5 Interpretability and Concept-Level Decoding

> Brain→Image，同时显式强调 **可解释性 / 概念层 / semantic bottleneck**。

MindReader: Reconstructing complex images from brain activities  
[[NeurIPS 2022](https://arxiv.org/abs/2209.12951)] [[Code](https://github.com/yuvalsim/MindReader)]

Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks  
[[NeurIPS 2025 Poster](https://openreview.net/forum?id=K6ijewH34E)] [[PDF](https://openreview.net/pdf?id=K6ijewH34E)]

---

### 4.6 Visual-to-fMRI Synthesis and Data Augmentation

> 方向反过来：**Image → fMRI**，但常用于合成 / 增广 fMRI，提升解码性能。

SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning  
[[arXiv 2025](https://arxiv.org/abs/2508.10298)] [[OpenReview](https://openreview.net/forum?id=ZTHYaSxqmq)]

*(Add more visual→fMRI encoders / synthesizers here.)*

---

## 5. Video and Dynamic Scene Decoding

> **Scope:** 输出是 **video / 动态帧 / 动态特征**，通常是 movie fMRI。

Visual experience reconstruction from movie fMRI  
[[Current Biology 2011](https://doi.org/10.1016/j.cub.2011.01.031)]

CLSR: Decoding complex video and story stimuli from fMRI  
[[Nature Neuroscience 2023](https://doi.org/10.1038/s41593-023-01327-2)]

*(Add movie fMRI → video / caption / dynamic scene decoding works here.)*

---

## 6. Multimodal and Foundation-Model-based Decoding

> **Scope:** 使用 CLIP / Stable Diffusion / VLM / LMM 等 **foundation models**，统一解多种模态（图像 + 文本等）的解码框架。

UMBRAE: Unified Multimodal Brain Decoding  
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf)] [[arXiv](https://arxiv.org/abs/2404.07202)] [[Project](https://weihaox.github.io/UMBRAE/)] [[Code](https://github.com/weihaox/UMBRAE)]

MindReader: Reconstructing complex images from brain activities  
[[NeurIPS 2022](https://arxiv.org/abs/2209.12951)] [[Code](https://github.com/yuvalsim/MindReader)]

*(More “unified multimodal” or foundation-model-centric decoders can be added here.)*

---

## 7. Audio and Music Decoding

> **Scope:** fMRI → 音频 / 音乐：音乐旋律、音色、自然声音类别、speech waveform 等。

*(Placeholder – add music / sound decoding works here.)*

---

## 8. Clinical / Cognitive and Mental-State Decoding

> **Scope:** fMRI 解码用于 emotion / cognitive load / disease marker / mental state 等临床或认知应用。

*(Placeholder – add clinical / cognitive decoding works here.)*

---

## 9. Toolboxes and Awesome Lists

> **Scope:** 通用代码库、预处理工具、以及其它 awesome 列表。

DeepImageReconstruction codebase  
[[GitHub](https://github.com/KamitaniLab/DeepImageReconstruction)]

semantic-decoding (language reconstruction)  
[[GitHub](https://github.com/HuthLab/semantic-decoding)]

MindEye2 implementation  
[[GitHub](https://github.com/MedARC-AI/MindEyeV2)]

Brain-Diffuser implementation  
[[GitHub](https://github.com/ozcelikfu/brain-diffuser)]

UMBRAE implementation  
[[GitHub](https://github.com/weihaox/UMBRAE)]

awesome-brain-decoding (general, multi-modality)  
[[GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)]

Awesome Brain Encoding & Decoding  
[[GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)]

Awesome Brain Graph Learning with GNNs  
[[GitHub](https://github.com/XuexiongLuoMQ/Awesome-Brain-Graph-Learning-with-GNNs)]

*(You can also add fMRIPrep, nilearn, visualization tools, etc.)*

---

## 10. Contributing

Contributions are welcome! 🎉  

**Recommended entry format:**

```markdown
Paper Title  
[[Venue Year](paper_link)] [[Code](code_link)] [[Project](project_link)] [[Dataset](dataset_link)]
