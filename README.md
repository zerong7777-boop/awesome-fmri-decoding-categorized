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

> **Scope:** Global surveys / reviews / tutorials that describe the overall landscape of fMRI decoding and brain-conditioned generative modeling (images, videos, language, multimodal, BCIs, etc.).

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[[arXiv 2025](https://arxiv.org/abs/2503.15978)]

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[[IEEE TAI 2025](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)] [[Project](https://github.com/MichaelMaiii/AIGC-Brain)]

Visual Image Reconstruction from Brain Activity via Latent Representation  
[[Annual Review of Vision Science 2025](https://www.annualreviews.org/content/journals/10.1146/annurev-vision-110423-023616)]

Non-Invasive Brain-Computer Interfaces: State of the Art and Trends  
[[IEEE Reviews in Biomedical Engineering 2025](https://doi.org/10.1109/RBME.2024.3449790)]

Progress, challenges and future of linguistic neural decoding with deep learning  
[[Communications Biology 2025](https://www.nature.com/articles/s42003-025-08511-z)]

Artificial intelligence based multimodal language decoding from brain activity: A review  
[[Brain Research Bulletin 2023](https://doi.org/10.1016/j.brainresbull.2023.110713)]

Review of visual neural encoding and decoding methods in fMRI  
[[Journal of Image and Graphics 2023](https://www.cjig.cn/en/article/doi/10.11834/jig.220525)]

Visualizing the mind’s eye: a future perspective on image reconstruction from brain signals  
[[Psychoradiology 2023](https://doi.org/10.1093/psyrad/kkad022)]

Deep learning approaches for neural decoding across multiple scales  
[[Briefings in Bioinformatics 2021](https://doi.org/10.1093/bib/bbaa053)]

A Survey on Brain Encoding and Decoding  
[[IJCAI 2021](https://www.ijcai.org/proceedings/2021/594)]

Deep Generative Models in Brain Encoding and Decoding  
[[Engineering 2019](https://doi.org/10.1016/j.eng.2019.03.011)]

---

## 2. Datasets and Benchmarks

> **Scope:** Public fMRI datasets / benchmarks (vision, language, audio, etc.), organized by dataset rather than method.

Natural Scenes Dataset (NSD) – 7T high-resolution fMRI responses to tens of thousands of natural images.  
[[Nature Neuroscience 2022](https://www.nature.com/articles/s41593-021-00962-x)] [[Website](https://naturalscenesdataset.org/)] [[Data](https://osf.io/9pjky/)]

Natural Object Dataset (NOD) – large-scale fMRI dataset with 57k naturalistic images (ImageNet / COCO) from 30 participants.  
[[Scientific Data 2023](https://www.nature.com/articles/s41597-023-02471-x)] [[OpenNeuro ds004496](https://openneuro.org/datasets/ds004496)]

THINGS-data / THINGS-fMRI – multimodal object-vision dataset (fMRI, MEG, behavior) over ~1.8k object concepts.  
[[eLife 2023](https://elifesciences.org/articles/82580)] [[OpenNeuro ds004192](https://openneuro.org/datasets/ds004192)] [[Collection](https://doi.org/10.25452/figshare.plus.c.6161151)]

BOLD5000 – slow event-related fMRI dataset for ~5k images drawn from COCO / ImageNet / SUN.  
[[Scientific Data 2019](https://www.nature.com/articles/s41597-019-0052-3)] [[Website](https://bold5000-dataset.github.io/website/)] [[OpenNeuro ds001499](https://openneuro.org/datasets/ds001499)]

Deep Image Reconstruction (DIR) dataset – single-subject fMRI for natural images used in the Kamitani deep image reconstruction work.  
[[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)] [[OpenNeuro ds001506](https://openneuro.org/datasets/ds001506)]

Narratives / Story listening datasets – multi-subject naturalistic spoken-story fMRI.  
[[Scientific Data 2021](https://www.nature.com/articles/s41597-021-01033-3)] [[Website](https://www.narrativeslab.org/)] [[OpenNeuro ds002345](https://openneuro.org/datasets/ds002345)]

Semantic reconstruction of continuous language – dataset used in the Nature Neuroscience 2023 semantic decoding paper.  
[[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9)] [[OpenNeuro ds003020](https://openneuro.org/datasets/ds003020)]

Natural language fMRI dataset for voxelwise encoding models – five multi-session natural-language listening datasets for voxelwise encoding.  
[[Scientific Data 2023](https://www.nature.com/articles/s41597-023-02437-z)] [[GitHub](https://github.com/HuthLab/deep-fMRI-dataset)]

BOLD Moments Dataset (BMD) – video fMRI responses to ~1k short naturalistic clips with rich object / action / text metadata.  
[[Nature Communications 2024](https://www.nature.com/articles/s41467-024-50310-3)] [[OpenNeuro ds005165](https://openneuro.org/datasets/ds005165)] [[Code](https://github.com/blahner/BOLDMomentsDataset)]

Spacetop – multimodal fMRI dataset with >100 participants, combining movie viewing with a broad battery of cognitive / affective tasks and physiology.  
[[Scientific Data 2025](https://www.nature.com/articles/s41597-025-05154-x)] [[OpenNeuro ds005256](https://openneuro.org/datasets/ds005256)]

Emo-FilM – film-based fMRI with dense emotion annotations and concurrent physiological recordings.  
[[Scientific Data 2025](https://www.nature.com/articles/s41597-025-04803-5)] [[OpenNeuro ds004892](https://openneuro.org/datasets/ds004892)]



---

## 3. Language / Narrative Decoding (Brain → Text)

> **Scope:** Methods that decode brain activity (primarily fMRI, plus a few closely related non-invasive signals) into **textual output**: words, sentences, story summaries, captions, etc.

Semantic reconstruction of continuous language from non-invasive brain recordings  
[[Nature Neuroscience 2023](https://www.nature.com/articles/s41593-023-01304-9)] [[Code](https://github.com/HuthLab/semantic-decoding)] [[Dataset](https://openneuro.org/datasets/ds003020)]

Generative language reconstruction from brain recordings (BrainLLM)  
[[Communications Biology 2025](https://www.nature.com/articles/s42003-025-07731-7)] [[Code](https://github.com/YeZiyi1998/Brain-language-generation)]

Language Generation from Human Brain Activities  
[[CoRR 2023](https://arxiv.org/abs/2311.09889)]

Decoding Continuous Character-based Language from Non-invasive Brain Recordings  
[[bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.03.19.585656v1)] [[arXiv](https://arxiv.org/abs/2403.11183)] [[Dataset](https://openneuro.org/datasets/ds006630)]

Brain-Inspired fMRI-to-Text Decoding via Incremental and Wrap-Up Language Modeling (CogReader)  
[[NeurIPS 2025 Spotlight](https://openreview.net/forum?id=REIo9ZLSYo)] [[PDF](https://openreview.net/pdf?id=REIo9ZLSYo)]

Open-vocabulary Auditory Neural Decoding Using fMRI-prompted LLM (Brain Prompt GPT / BP-GPT)  
[[ACM MM 2024](https://arxiv.org/abs/2405.07840)] [[PDF](https://arxiv.org/pdf/2405.07840.pdf)]

Language Reconstruction with Brain Predictive Coding from fMRI Data (PredFT)  
[[arXiv 2024; submitted to ICLR 2025](https://arxiv.org/abs/2405.11597)]

MapGuide: A Simple yet Effective Method to Reconstruct Continuous Language from Brain Activities  
[[NAACL 2024](https://aclanthology.org/2024.naacl-long.211/)] [[PDF](https://aclanthology.org/2024.naacl-long.211.pdf)]

MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding  
[[arXiv 2025](https://arxiv.org/abs/2502.15786)]

Towards decoding individual words from non-invasive brain recordings *(EEG/MEG – non-fMRI but highly influential for non-invasive brain-to-text)*  
[[Nature Communications 2025](https://www.nature.com/articles/s41467-025-65499-0)]


---

## 4. Visual Image Reconstruction (Brain → Image)

> **Scope:** Methods that reconstruct **static images** from brain activity (typically fMRI). Sub-categories 4.1–4.6 group works by the main generative backbone (classical / GAN / diffusion) or by a key modeling focus (cross-subject generalization, interpretability, visual-to-fMRI synthesis).

### 4.1 Classical and Pre-Generative

> Early approaches that do **not** rely on modern deep generative image models (GAN / diffusion).

Visual image reconstruction from human brain activity using a combination of multiscale local image decoders  
[[Neuron 2008](https://doi.org/10.1016/j.neuron.2008.11.004)]

Reconstructing Natural Scenes from fMRI Patterns using Hierarchical Visual Features  
[[NeuroImage 2011](https://doi.org/10.1016/j.neuroimage.2010.07.063)]

---

### 4.2 GAN / VAE-based

> Generative backbone is mainly **GAN / VAE / convnets**, rather than diffusion.

Deep image reconstruction from human brain activity  
[[PLoS Comput Biol 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)] [[Code](https://github.com/KamitaniLab/DeepImageReconstruction)] [[Dataset](https://openneuro.org/datasets/ds001506)]

From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI  
[[NeurIPS 2019](https://arxiv.org/abs/1907.02431)] [[Code](https://github.com/WeizmannVision/ssfmri2im)]

Reconstructing Natural Scenes from fMRI Patterns using BigBiGAN  
[[IJCNN 2020](https://arxiv.org/abs/2011.12243)]

---

### 4.3 Diffusion-based Reconstruction

> fMRI→image methods that use **diffusion / latent diffusion / Stable Diffusion** as the main generative prior.

High-resolution image reconstruction with latent diffusion models from human brain activity  
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html)] [[Project](https://sites.google.com/view/stablediffusion-with-brain/)] [[Code](https://github.com/yu-takagi/StableDiffusionReconstruction)]

Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding (MinD-Vis)  
[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.pdf)] [[Project](https://mind-vis.github.io/)]

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)  
[[NeurIPS 2023](https://arxiv.org/abs/2305.18274)] [[Project](https://medarc-ai.github.io/mindeye/)] [[Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)]

Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion  
[[Scientific Reports 2023](https://www.nature.com/articles/s41598-023-42891-8)] [[arXiv](https://arxiv.org/abs/2303.05334)] [[Code](https://github.com/ozcelikfu/brain-diffuser)]

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion  
[[ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3613819)] [[arXiv](https://arxiv.org/abs/2308.04249)] [[Code](https://github.com/YingxingLu/MindDiffuser)]

Dual-Guided Brain Diffusion Model: Natural Image Reconstruction from Human Visual Stimulus fMRI (DBDM)  
[[Bioengineering 2023](https://www.mdpi.com/2306-5354/10/10/1117)]

Mental image reconstruction from human brain activity  
[[Neural Networks 2024](https://www.sciencedirect.com/science/article/pii/S0893608023006470)]

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction  
[[IEEE TIP 2025](https://ieeexplore.ieee.org/document/10749645)] [[arXiv](https://arxiv.org/abs/2401.01713)]

Balancing Semantic and Structural Decoding for fMRI-to-Image Reconstruction  
[[Expert Systems with Applications 2025](https://www.sciencedirect.com/science/article/abs/pii/S0957417425034517)]

---

### 4.4 Cross-Subject and Generalizable Decoding

> Still Brain→Image, but emphasizing **cross-subject generalization**, few-shot adaptation, mixture-of-experts, and shared-subject models.

MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data  
[[ICML 2024](https://proceedings.mlr.press/v235/scotti24a.html)] [[arXiv](https://arxiv.org/abs/2403.11207)] [[Project](https://medarc-ai.github.io/mindeye2/)] [[Code](https://github.com/MedARC-AI/MindEyeV2)]

ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding  
[[OpenReview](https://openreview.net/pdf?id=7a4f583ef54685490be5c58986a3ad803aac087c)] [[Code](https://github.com/xmed-lab/ZEBRA)]

Psychometry: An Omnifit Model for Image Reconstruction from Human Brain Activity  
[[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Quan_Psychometry_An_Omnifit_Model_for_Image_Reconstruction_from_Human_Brain_CVPR_2024_paper.html)] [[arXiv](https://arxiv.org/abs/2403.20022)]

NeuroPictor: Refining fMRI-to-Image Reconstruction via Multi-individual Pretraining and Multi-level Modulation  
[[ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06746.pdf)] [[arXiv](https://arxiv.org/abs/2403.18211)] [[Project](https://jingyanghuo.github.io/neuropictor/)]

Wills Aligner: Multi-Subject Collaborative Brain Visual Decoding  
[[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33554)] [[arXiv](https://arxiv.org/abs/2404.13282)]

BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities  
[[AAAI 2025 Oral](https://ojs.aaai.org/index.php/AAAI/article/view/33579)] [[arXiv](https://arxiv.org/abs/2501.14309)] [[Project](https://github.com/kunzhan/brainguard)]

MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding  
[[OpenReview](https://openreview.net/forum?id=fYSPRGmS6l)] [[arXiv](https://arxiv.org/abs/2505.15946)] [[Code](https://github.com/yuxiangwei0808/MoRE-Brain)]

---

### 4.5 Interpretability and Concept-Level Decoding

> Brain→Image methods that explicitly emphasize **interpretability, concept-level representations, semantic bottlenecks, or analyzing what generative priors really use from the brain**.

MindReader: Reconstructing complex images from brain activities  
[[NeurIPS 2022](https://arxiv.org/abs/2209.12951)] [[Code](https://github.com/yuvalsim/MindReader)]

Bridging Brains and Concepts: Interpretable Visual Decoding from fMRI with Semantic Bottlenecks  
[[NeurIPS 2025 Poster](https://openreview.net/forum?id=K6ijewH34E)] [[PDF](https://openreview.net/pdf?id=K6ijewH34E)]

BrainBits: How Much of the Brain are Generative Reconstruction Methods Using?  
[[NeurIPS 2024](https://openreview.net/forum?id=KAAUvi4kpb)] [[arXiv](https://arxiv.org/abs/2411.02783)] [[Code](https://github.com/czlwang/BrainBits)]

---

### 4.6 Visual-to-fMRI Synthesis and Data Augmentation

> Reverse direction (**image → fMRI / encoding**), often used to **synthesize / augment fMRI data**, or to build universal encoders that support better decoding.

Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification from Brain Activity  
[[NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S105381192200249X)]

The Wisdom of a Crowd of Brains: A Universal Brain Encoder  
[[arXiv 2024](https://arxiv.org/abs/2406.12179)]

SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning  
[[arXiv 2025](https://arxiv.org/abs/2508.10298)] [[OpenReview](https://openreview.net/forum?id=ZTHYaSxqmq)]


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
