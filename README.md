# Awesome fMRI Decoding (Categorized)
A curated list of **fMRI-based brain decoding** papers and resources, focusing on **reconstructing images / videos / language / audio** from brain activity, inspired by [awesome-diffusion-categorized](https://github.com/wangkai930418/awesome-diffusion-categorized).

If you find this repo helpful, please give it a ⭐ and consider contributing new papers via pull requests.

---

## Contents

- [Survey & Overview](#survey--overview)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Language / Narrative Decoding (Brain → Text)](#language--narrative-decoding-brain--text)
- [Visual Image Reconstruction (Brain → Image)](#visual-image-reconstruction-brain--image)
  - [Pre-generative / Early Works](#pre-generative--early-works)
  - [GAN / VAE based](#gan--vae-based)
  - [Diffusion-based](#diffusion-based)
- [Video & Dynamic Scene Decoding](#video--dynamic-scene-decoding)
- [Audio & Music Decoding](#audio--music-decoding)
- [Multimodal & Foundation-Model-based Decoding](#multimodal--foundation-model-based-decoding)
- [Clinical / Cognitive & Mental-State Decoding](#clinical--cognitive--mental-state-decoding)
- [Toolboxes & Tutorials](#toolboxes--tutorials)
- [Contributing](#contributing)
- [License](#license)

---

## Survey & Overview

A Survey on fMRI-based Brain Decoding for Reconstructing Multimodal Stimuli  
[ArXiv 2025] [Paper](https://arxiv.org/abs/2503.15978) [Project](https://github.com/LpyNow/BrainDecodingImage)  

Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy  
[IEEE TAI 2025] [Paper](https://www.computer.org/csdl/journal/ai/2025/05/10798967/22EatqRGQxO)  

Deep Generative Models in Brain Encoding and Decoding (review)  
[Engineering 2019] [Paper](https://doi.org/10.1016/j.eng.2019.03.011)  

---

## Datasets & Benchmarks

Natural Scenes Dataset (NSD)  
[NeuroImage / Sci Data] [Website](https://naturalscenesdataset.org/) [Data](https://osf.io/9pjky/) [Image, I2I/I2T]

Deep Image Reconstruction (DIR) dataset  
[PLoS Comput Biol 2019] [Data](https://openneuro.org/datasets/ds001506) [Image, I2I]

Narratives / Story listening datasets  
[NeuroImage / Sci Data] [Narratives](https://www.narrativeslab.org/) [Data](https://openneuro.org/datasets/ds002345) [Sound, S2S/S2T]

Semantic reconstruction of continuous language from non-invasive brain recordings – dataset  
[Nat Neurosci 2023] [Data (OpenNeuro)](https://openneuro.org/datasets/ds003020) [Sound, S2T]

(Feel free to add more: Vim-1, BOLD5000, GOD, CelebrityFace, etc.)

---

## Language / Narrative Decoding (Brain → Text)

Semantic reconstruction of continuous language from non-invasive brain recordings  
[Nat Neurosci 2023] [Paper](https://www.nature.com/articles/s41593-023-01304-9) [Code](https://github.com/HuthLab/semantic-decoding) [Dataset](https://openneuro.org/datasets/ds003020) [S2T, fMRI]

(Add more narrative / language decoding works here.)

---

## Visual Image Reconstruction (Brain → Image)

### Pre-generative / Early Works

Reconstructing Natural Scenes from fMRI Patterns using Hierarchical Visual Features  
[NeuroImage 2011] [Paper](https://doi.org/10.1016/j.neuroimage.2010.07.063) [I2I]

### GAN / VAE based

Deep image reconstruction from human brain activity  
[PLoS Comput Biol 2019] [Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633) [Code](https://github.com/KamitaniLab/DeepImageReconstruction) [Dataset](https://openneuro.org/datasets/ds001506) [I2I, GAN/VAE, Image]

From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI  
[NeurIPS 2019] [Paper](https://arxiv.org/abs/1907.02431) [Code](https://github.com/WeizmannVision/ssfmri2im) [I2I, Self-supervised, Image]

Reconstructing Natural Scenes from fMRI Patterns using BigBiGAN  
[Preprint] [Paper](https://www.researchgate.net/publication/347156512_Reconstructing_Natural_Scenes_from_fMRI_Patterns_using_BigBiGAN) [I2I, BigGAN]

### Diffusion-based

Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion  
[Sci Rep 2023] [Paper](https://www.nature.com/articles/s41598-023-42891-8) [Code](https://github.com/ozcelikfu/brain-diffuser) [Dataset: NSD](https://naturalscenesdataset.org/) [I2I, Diffusion]

Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors (MindEye)  
[NeurIPS 2023] [Paper](https://arxiv.org/abs/2305.18274) [Project](https://medarc-ai.github.io/mindeye/) [Code](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) [I2I, Retrieval + Diffusion]

MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion  
[ACM MM 2023] [Paper](https://arxiv.org/abs/2308.04249) [Code](https://github.com/YingxingLu/MindDiffuser) [I2I, Diffusion]

NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction  
[IEEE TIP 2025] [Paper](https://arxiv.org/abs/2401.01713) [Code](https://github.com/neu-diffusion/NeuralDiffuser) [I2I, Diffusion]

Mental image reconstruction from human brain activity  
[Neural Networks 2024] [Paper](https://www.sciencedirect.com/science/article/pii/S0893608023006470) [I2I, Diffusion-guided]

MindEye2: Shared-subject models enable fMRI-to-image with one hour of data  
[Preprint / 2024] [Project](https://medarc-ai.github.io/mindeye2/) [I2I, Diffusion, Shared-subject]

(You can continue adding more diffusion-based works: BrainSD, MinD-Vis, MindReader, UMBRAE, SynBrain, etc.)

---

## Video & Dynamic Scene Decoding

Visual experience reconstruction from movie fMRI  
[Current Biology 2011] [Paper](https://doi.org/10.1016/j.cub.2011.01.031) [V2V]

CLSR: Decoding complex video and story stimuli from fMRI  
[Nat Neurosci 2023] [Paper](https://doi.org/10.1038/s41593-023-01327-2) [V2V/V2T]

(Place movie / video decoding and V2T works here.)

---

## Audio & Music Decoding

Music-evoked fMRI datasets and decoding of music genre / affect  
[Various years] (e.g., MusicAffect, GTZan-fMRI)  

(You can add papers on music genre / affect decoding, sound category decoding, etc.)

---

## Multimodal & Foundation-Model-based Decoding

MindReader: Reconstructing complex images from brain activities  
[NeurIPS 2022] [Paper](https://arxiv.org/abs/2209.12951) [Code](https://github.com/yuvalsim/MindReader) [I2I, CLIP, StyleGAN2]

UMBRAE: Unified Multimodal Brain Decoding  
[ECCV 2024] [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01133.pdf) [Project](https://weihaox.github.io/UMBRAE) [I2I/I2T, Multimodal]

Brain-conditional multimodal synthesis works (text, audio, video conditioned on fMRI)  
(Place here: e.g., BrainCaptioning, AIGC-Brain style collections.)

---

## Clinical / Cognitive & Mental-State Decoding

(Reserved for works that use fMRI decoding for clinical, cognitive or mental-state tasks, e.g., decoding emotion, disease markers, etc.)

---

## Toolboxes & Tutorials

DeepImageReconstruction codebase  
[Code](https://github.com/KamitaniLab/DeepImageReconstruction) – End-to-end pipeline for visual fMRI → image reconstruction.

awesome-brain-decoding (general, multi-modality)  
[GitHub](https://github.com/NeuSpeech/awesome-brain-decoding)

Awesome Brain Encoding & Decoding  
[GitHub](https://github.com/subbareddy248/Awesome-Brain-Encoding--Decoding)

(You can also list preprocessing toolboxes, convenient fMRI loaders, and educational tutorials.)

---

## Contributing

Contributions are welcome!  
If you want to add or update a paper:

1. Check whether it is **fMRI-based decoding** (or at least has a strong fMRI decoding component).
2. Choose the right category (and subcategory, if applicable).
3. Follow the existing format:

   ```markdown
   Paper Title  
   [Conf/Journal Year] [Paper](...) [Code](...) [Dataset](...) [Task tags] [Model tags]
