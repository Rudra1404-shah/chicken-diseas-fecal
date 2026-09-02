# 🐔 Chicken Disease Classifier

Binary classification model for detecting diseases in poultry using deep learning and transfer learning on fecal images.

## 📋 Overview

This project builds a **binary classification model** to distinguish between healthy and diseased poultry based on fecal images. The classifier uses **transfer learning with EfficientNetB0** trained on 8,067 images collected from Tanzania, achieving **95% accuracy**.

### Problem Statement

Newcastle disease and coccidiosis spread through poultry flocks within days. Smallholder farmers in rural areas lack veterinary access, and by the time visible symptoms appear, the disease is already widespread.

**Key Insight:** Droppings change color and texture early—before the bird looks sick. A phone photo is the cheapest diagnostic signal available on a small farm.

---

## 📊 Dataset

| Class | Count | Details |
|-------|-------|---------|
| Healthy | 2,404 | Baseline, no disease |
| Coccidiosis | 2,476 | Parasitic infection |
| Salmonella | 2,625 | Bacterial infection |
| Newcastle Disease | 562 | Most lethal, smallest class |
| **Total** | **8,067** | **Collapsed to binary: Healthy (0) vs Disease (1)** |

**Source:** Poultry fecal images, Tanzania  
**Class Balance:** 70% diseased, 30% healthy (imbalanced)  
**Original Labels:** 4-way classification → Converted to binary for this project

### ⚠️ Data Leakage Fix

**The Problem:**  
Images were photographed multiple times (seconds apart). Random train/test split would scatter near-duplicate photos on both sides, letting the model memorize rather than generalize.

**The Solution:**
1. **Perceptual hashing** — Convert each image to 8×8 grayscale, extract 64-bit hash
2. **Clustering** — Group images with matching hashes
3. **StratifiedGroupKFold** — Keep entire clusters on one side of split

**Result:** Honest validation metrics measuring true generalization, not memory.

---

## 🏗️ Model Architecture

### Why EfficientNetB0?

With only 8,067 images, training from scratch is impossible. **Transfer learning** provides the largest accuracy boost.

| Model | Parameters | Choice | Reason |
|-------|-----------|--------|--------|
| VGG16 | 138M | ❌ Too large | 25× bigger for worse accuracy |
| **EfficientNetB0** | **5.3M** | ✅ **Chosen** | Best accuracy-per-FLOP; container-friendly |
| Baseline CNN | 1.2M | Baseline | Trained from scratch; establishes floor |

### Training Pipeline

**Phase 1: Head Training (Backbone Frozen)**
```
Optimizer:     Adam, lr = 1e-3
Epochs:        15 max
Strategy:      Train random head before backprop
Reason:        Prevents catastrophic forgetting
```

**Phase 2: Fine-tuning (Top 30% Unfrozen)**
```
Optimizer:     Adam, lr = 1e-5
Epochs:        25 max
Strategy:      Re-learn late-layer ImageNet semantics
Reason:        'dog ear' → 'abnormal fecal texture'
```

**Constants:**
- ✅ BatchNorm frozen
- ✅ Class weights applied (70% diseased)
- ✅ Early stopping on val AUC

---

## 📈 Results

**Test Set:** 1,138 held-out images

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| ROC-AUC | 0.935 |
| Error Reduction | 72% (vs baseline) |

### Transfer Learning Impact

```
Baseline CNN (from scratch):    AUC = 0.8854
EfficientNetB0 (transfer):      AUC = 0.9354
─────────────────────────────────────────────
Improvement:                    +72% error reduction
```

### Per-Class Performance

| Disease | Recall | Notes |
|---------|--------|-------|
| Healthy | 96.2% | High specificity |
| Coccidiosis | 93.4% | Well-represented |
| Salmonella | 94.8% | Balanced class |
| Newcastle | 81.9% | ⚠️ Weakest (only 562 images) |

---

## 📁 Project Structure

```
chicken-disease-classifier/
├── Diseas Classification.ipynb    # Main notebook (EDA + training)
├── main.py                        # Inference/prediction script
├── train_data.csv                 # Training dataset
├── Exam Test/                     # Test images
├── ccd/                           # Additional data/code
├── chicken-disease-presentation.pptx  # Full presentation
└── README.md                      # This file
```

### Files Description

#### `Diseas Classification.ipynb`
- ✅ Exploratory Data Analysis (EDA)
- ✅ Perceptual hashing & deduplication
- ✅ Data split with StratifiedGroupKFold
- ✅ Two-phase training pipeline
- ✅ Model evaluation & metrics
- ✅ Visualization & insights

**How to Run:**
```bash
jupyter notebook "Diseas Classification.ipynb"
```

#### `main.py`
Python script for:
- Loading trained model
- Predicting on single or batch images
- Generating reports

**Usage:**
```bash
python main.py --image path/to/image.jpg
python main.py --batch path/to/folder/
```

#### `train_data.csv`
Dataset metadata (image paths, labels, hashes)

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Rudra1404-shah/binary-classification-project-chicken.git
cd binary-classification-project-chicken
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- TensorFlow 2.13+
- Keras 3.10+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Pillow (image processing)
- Jupyter (for notebook)

### 3. Run Notebook
```bash
jupyter notebook "Diseas Classification.ipynb"
```

### 4. Make Predictions
```bash
python main.py --image "Exam Test/healthy_sample.jpg"
```

**Output:**
```
Prediction: Healthy
Probability: 0.96
Confidence: High
```

---

## 💻 Usage Examples

### Load Trained Model & Predict

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model_efficientnetb0.h5')

# Load & preprocess image
img = Image.open('path/to/image.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, 0)

# Predict
prediction = model.predict(img_batch)
label = 'Diseased' if prediction[0][0] > 0.5 else 'Healthy'
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Label: {label}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Prediction

```bash
python main.py --batch "path/to/test_images/" --output predictions.csv
```

---

## ⚠️ Limitations & Known Issues

### 1. Newcastle Disease Recall: 81.9%
The disease that spreads fastest is hardest to detect. Only 562 training images—too few for robust learning.

**Recommendation:** Collect 2,000+ Newcastle samples before production.

### 2. Near-Duplicate Images
Only exact hash matches are grouped. Images with slight differences (crop, compression) can still leak across train/test.

**Full pairwise comparison** is O(n²) and computationally expensive.

### 3. Single-Source Data
- **Geography:** Only two Tanzania regions
- **Collection Window:** Single timeframe
- **Camera:** One camera app

**Domain shift to other farms/regions is untested.**

### 4. Imbalanced Classes
70% diseased vs 30% healthy. Model may prioritize majority class without proper weighting.

---

## 🔮 Future Improvements

- [ ] Collect 2,000+ Newcastle disease samples
- [ ] Validate on data from Kenya, Uganda, Rwanda
- [ ] Uncertainty quantification (Bayesian dropout)
- [ ] Explainability (Grad-CAM for lesion localization)
- [ ] Multi-class output (specific disease detection)
- [ ] Model compression (TensorFlow Lite for mobile)
- [ ] REST API deployment (FastAPI)
- [ ] Real-time mobile app (Flutter/React Native)

---

## 📚 Technical Details

### Class Imbalance Handling

**Method:** Class weights + SMOTE  
**Rationale:** 70% diseased class → model learns minority patterns  
**Metric:** Optimized for **recall** (catch max sick birds), not accuracy

### Validation Strategy

**StratifiedGroupKFold (not random split)**
- Prevents data leakage
- Honors group structure
- Gives honest generalization estimate

### Hardware Requirements

- **Training:** GPU recommended (NVIDIA Tesla A100 / RTX 3090)
- **Inference:** CPU only (no GPU needed)
- **RAM:** 16GB+ for notebook execution
- **Disk:** 2GB+ (model + data)

---

## 📊 Presentation

Full project presentation available: **`chicken-disease-presentation.pptx`**

Covers:
- Problem statement
- Dataset overview
- Data leakage fix (methodology)
- Architecture decisions
- Training pipeline
- Results & metrics
- Deployment considerations
- Limitations & future work

---

## 🤝 How to Use This Repository

1. **For Learning:** Read the notebook end-to-end (`Diseas Classification.ipynb`)
2. **For Inference:** Use `main.py` on your own fecal images
3. **For Reproduction:** Follow notebook steps to retrain on `train_data.csv`
4. **For Extension:** Modify architecture/hyperparams and run `main.py` again

---

## 📝 Key Takeaways

✅ **Transfer learning** beat from-scratch training by 72% error reduction  
✅ **Data leakage fix** (perceptual hashing) ensures honest metrics  
✅ **Class imbalance** handled with weights + SMOTE  
✅ **Real-world constraint:** Small model fits on cheap hardware  
✅ **Honest evaluation:** Newcastle recall still only 81.9% (not hiding weaknesses)  

---

## 📚 References

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Best Practices](https://cs231n.github.io/transfer-learning/)
- [Class Imbalance Handling](https://imbalanced-learn.org/stable/)
- [Perceptual Hashing](https://www.phash.org/)

---

## 📧 Questions?

- **Training questions:** See `Diseas Classification.ipynb`
- **Inference questions:** Check `main.py` code comments
- **Data questions:** Review `train_data.csv` structure
- **Issues:** Open a GitHub Issue

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file

---

**Built for smallholder farmers | Semester 6 Project | RUDRA SHAH**

*Last Updated: November 2024*# binary-classification-project-chicken
