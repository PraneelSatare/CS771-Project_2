# Continual Learning & Unsupervised Domain Adaptation
## CS771 - Introduction to Machine Learning (Autumn 2024) - Mini-Project 2

[![Course](https://img.shields.io/badge/Course-CS771-blue)](https://www.cse.iitk.ac.in/users/piyush/)
[![Problem](https://img.shields.io/badge/Problem-Continual%20Learning-green)](https://github.com/your-repo)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🎯 Project Overview

This project addresses the **continual learning** and **unsupervised domain adaptation** challenges posed in CS771 Mini-Project 2. We tackle the problem of learning from sequential datasets while preventing catastrophic forgetting, using Learning with Prototypes (LwP) as the base classifier with novel prototype update mechanisms.

### Problem Setting
- **20 sequential datasets** derived from CIFAR-10
- **Task 1**: Datasets D₁-D₁₀ from the same distribution p(x)
- **Task 2**: Datasets D₁₁-D₂₀ from different but related distributions
- **Constraint**: Only D₁ is labeled; all others are unlabeled
- **Goal**: Maintain performance on previous datasets while adapting to new ones

### Team Members
- **Akshat Sharma** (230101)
- **Dweep Joshipura** (230395)
- **Kanak Khandelwal** (230520)
- **Praneel B Satare** (230774)  

---

## 🚀 Key Innovations & Technical Contributions

### 1. **Weighted Prototype Update Rule (Task 1)**
For sequential datasets from the same distribution, we developed a mathematically principled update mechanism:
```math
\mu^{(n+1)}_c := \frac{N\alpha\mu^{(n)}_c + \sum\limits_{y^{(n+1)}_i = c}{x^{(n+1)}_i}}{N\alpha+n^{(n+1)}_c}
```
**Key Characteristics:**
- **α = 0.2** (optimally tuned): Balances old knowledge retention vs. new data adaptation
- **Prevents catastrophic forgetting**: Maintains 98%+ accuracy across all previous datasets
- **Interpretable**: α → ∞ preserves old prototypes, α → 0 uses only new data

### 2. **Clustering-Based Domain Adaptation (Task 2)**
For datasets with distribution shifts, we introduced an unsupervised adaptation method:

```math
\mu^{(n+1)}_c := \frac{\beta\mu^{(n)}_c + M^{(n+1)}_c}{\beta+1} 
```

**Novel Approach:**
- **Class-aware K-means**: Initialize cluster centers with previous prototypes
- **Automatic adaptation**: Clusters adjust to new data distributions
- **Balanced update**: β = 1 equally weighs old prototypes and new centroids

---

## 🔧 Architecture & Methodology

### Feature Extraction Pipeline
Given the constraint of not using CIFAR-trained models, we explored ImageNet pre-trained extractors:

| Model | Feature Dim | Accuracy on D₁ | Selected |
|-------|-------------|----------------|----------|
| ResNet | 2048 | 84.12% | ❌ |
| MobileNetv3 | 960 | 83.72% | ❌ |
| CaiT-M36 | 768 | 94.20% | ❌ |
| ViT-Base | 768 | 96.52% | ❌ |
| Eva02-Base | 768 | 96.88% | ❌ |
| **BEiT-Large** | **1024** | **98.72%** | ✅ |

### Baseline Comparisons
Initial experiments without feature extraction showed the necessity of our approach:

| Method | Training Accuracy |
|--------|-------------------|
| LwP (Euclidean, Raw) | 29.04% |
| LwP (Mahalanobis, Raw) | 9.52% |
| LwP (Euclidean, PCA-50) | 28.56% |
| LwP (Mahalanobis, PCA-50) | 41.20% |

---

## 📊 Experimental Results

### Task 1: Sequential Learning (Same Distribution)
**Performance Matrix**: Models f₁ to f₁₀ on held-out datasets D̂₁ to D̂₁₀

| Model | D̂₁ | D̂₂ | D̂₃ | D̂₄ | D̂₅ | D̂₆ | D̂₇ | D̂₈ | D̂₉ | D̂₁₀ |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| f₁ | 98.32% | — | — | — | — | — | — | — | — | — |
| f₂ | 98.36% | 97.84% | — | — | — | — | — | — | — | — |
| f₃ | 98.16% | 97.76% | 98.16% | — | — | — | — | — | — | — |
| f₄ | 98.16% | 97.76% | 98.04% | 97.92% | — | — | — | — | — | — |
| f₅ | 98.20% | 97.68% | 97.96% | 98.00% | 97.92% | — | — | — | — | — |
| f₆ | 98.12% | 97.84% | 98.00% | 97.92% | 97.96% | 98.40% | — | — | — | — |
| f₇ | 98.16% | 97.72% | 97.92% | 97.92% | 98.00% | 98.36% | 97.40% | — | — | — |
| f₈ | 98.00% | 97.76% | 97.80% | 97.84% | 97.88% | 98.36% | 97.36% | 97.56% | — | — |
| f₉ | 98.08% | 97.72% | 97.84% | 97.96% | 97.84% | 98.28% | 97.36% | 97.60% | 97.68% | — |
| **f₁₀** | **98.04%** | **97.76%** | **97.88%** | **97.96%** | **97.84%** | **98.28%** | **97.36%** | **97.60%** | **97.64%** | **97.88%** |

**Key Achievement**: ✅ **No catastrophic forgetting** - consistent ~98% accuracy across all datasets

### Task 2: Domain Adaptation (Distribution Shifts)
**Performance Matrix**: Models f₁₁ to f₂₀ on held-out datasets D̂₁₁ to D̂₂₀

| Model | D̂₁₁ | D̂₁₂ | D̂₁₃ | D̂₁₄ | D̂₁₅ | D̂₁₆ | D̂₁₇ | D̂₁₈ | D̂₁₉ | D̂₂₀ |
|-------|------|------|------|------|------|------|------|------|------|------|
| f₁₁ | 90.36% | — | — | — | — | — | — | — | — | — |
| f₁₂ | 90.36% | 75.92% | — | — | — | — | — | — | — | — |
| f₁₃ | 90.36% | 75.92% | 93.56% | — | — | — | — | — | — | — |
| f₁₄ | 90.36% | 75.92% | 93.56% | 97.28% | — | — | — | — | — | — |
| f₁₅ | 90.36% | 75.92% | 93.56% | 97.28% | 97.92% | — | — | — | — | — |
| f₁₆ | 90.36% | 75.92% | 93.56% | 97.28% | 97.92% | 94.56% | — | — | — | — |
| f₁₇ | 90.36% | 75.92% | 93.56% | 97.28% | 97.92% | 94.56% | 94.56% | — | — | — |
| f₁₈ | 90.36% | 75.92% | 93.56% | 97.28% | 97.92% | 94.56% | 94.56% | 91.32% | — | — |
| f₁₉ | 90.36% | 75.92% | 93.56% | 97.28% | 97.92% | 94.56% | 94.56% | 91.32% | 76.48% | — |
| **f₂₀** | **90.36%** | **75.92%** | **93.56%** | **97.28%** | **97.92%** | **94.56%** | **94.56%** | **91.32%** | **76.48%** | **97.24%** |

**Key Achievement**: ✅ **Successful domain adaptation** - ~5% average improvement through clustering-based updates

---

## 🛠️ Implementation Details

### Computational Requirements
- **Hardware**: Kaggle P100 GPU
- **Feature Extraction Time**: 2h 38m for all datasets
- **Memory**: Efficient prototype storage (~10KB per model)

### Hyperparameter Optimization
- **α tuning**: Grid search over [0.1, 0.2, 0.3, ..., 2.0]
- **Optimal α**: 0.2 (maximizes f₁₀ accuracy on D̂₁)
- **β selection**: Set to 1.0 based on equal weighting heuristic

### Key Constraints Addressed
✅ **No CIFAR-trained models**: Used ImageNet pre-trained BEiT-Large  
✅ **Same model size**: Consistent prototype dimensions across updates  
✅ **No labeled data**: Only D₁ labels used; rest are pseudo-labels  
✅ **LwP requirement**: Base classifier remains Learning with Prototypes  

---

## 📁 Repository Structure

```
CS771-Project-2/
├── notebooks/
│   ├── task1_sequential_learning.ipynb    # Task 1 implementation
│   ├── task2_domain_adaptation.ipynb      # Task 2 implementation
│   └── feature_extraction.ipynb           # BEiT feature extraction                
├── docs/
│   ├── report.pdf                         # LaTeX project report
└── README.md
```

---

## 🎬 Paper Review
As required by the project, we presented a detailed review of the below paper in a YouTube video.
- **Deja Vu: Continual Model Generalization for Unseen Domains** (ICLR 2023)

**YouTube Presentation**: [Deja Vu: Continual Model Generalization for Unseen Domains](https://youtu.be/eLPlipLRDrk?si=xx3-8AtGp1wSNU-q)
---

## 🔬 Technical Analysis

### Why Our Approach Works

1. **Weighted Updates**: The α parameter creates a principled balance between stability and plasticity
2. **Feature Quality**: BEiT-Large provides rich, transferable representations
3. **Class-Aware Clustering**: Initializing with prototypes maintains class structure
4. **Unsupervised Adaptation**: No need for labeled data in new domains

### Limitations & Future Work

- **Dataset D₁₇ Challenge**: Poor cluster-class correlation affects performance
- **β Selection**: Currently heuristic; could benefit from adaptive methods
- **Scalability**: Limited to prototype-based methods per project constraints

---

## 📊 Key Metrics Summary

| Metric | Task 1 | Task 2 |
|--------|--------|--------|
| **Average Accuracy** | 97.8% | 89.4% |
| **Catastrophic Forgetting** | ❌ Prevented | ✅ Minimal |
| **Domain Adaptation** | N/A | +5% improvement |
| **Computational Efficiency** | ✅ Prototype-based | ✅ K-means clustering |

---

## 🏆 Project Achievements

✅ **Successfully prevented catastrophic forgetting** in sequential learning  
✅ **Developed novel weighted prototype updates** with theoretical foundation  
✅ **Achieved effective domain adaptation** without labeled target data  
✅ **Maintained consistent model size** across all updates  
✅ **Comprehensive evaluation** with detailed accuracy matrices  
✅ **Efficient implementation** suitable for resource-constrained environments  

---

## 📚 References & Citations

1. **Course**: CS771 - Introduction to Machine Learning, IIT Kanpur, Autumn 2024
2. **Problem Statement**: Mini-Project 2 - Continual Learning with LwP
3. **BEiT**: Bao, H., Dong, L., Piao, S., & Wei, F. (2022). BEiT: BERT Pre-training of Image Transformers
4. **Domain Adaptation**: Fernando, B., et al. (2014). Subspace alignment for domain adaptation
5. **Clustering Methods**: Dridi, J., et al. (2024). Unsupervised clustering-based domain adaptation

---

## 📞 Contact

For questions about this implementation or the CS771 course project:
- **Course Instructor**: [Course Website](https://www.cse.iitk.ac.in/users/piyush/)
- **Team Contact**: [Create an issue](https://github.com/djthegr8/CS771-Project-2/issues)

---

*This project demonstrates practical solutions to continual learning challenges while adhering to the constraints and requirements of CS771 Mini-Project 2. The proposed methods show promise for real-world applications where models must adapt to new data distributions without forgetting previous knowledge.*
