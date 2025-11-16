# 🦷 Teeth Classification using Deep Learning

## 📌 Project Overview
This project presents a *comprehensive computer vision solution* for *teeth classification*.  
We experimented with multiple pretrained architectures as well as a CNN model from scratch. The best-performing model, **ResNet50**, was deployed using Streamlit for interactive predictions.

The goal is to preprocess and visualize dental images, then build and train a robust deep learning model to classify teeth into **7 distinct categories**.

Accurate teeth classification supports:
- 🏥 Enhanced diagnostic precision  
- 🤖 AI-driven dental applications  
- 😀 Improved patient outcomes in healthcare  

---

## 🚀 Goals & Models
**Objective:** Classify medical images into 7 dental disease categories:  
`CaS, CoS, Gum, MC, OC, OLP, OT`

**Models Applied:**
- 🧩 **ResNet50** ✅ (Best performing, deployed)  
- 🔬 DenseNet121  
- 📱 MobileNet  
- 🧠 Vision Transformer (ViT)  
- 🏗️ CNN from scratch  

**Deployment:**
- Implemented using **Streamlit**  
- Model weights stored on Google Drive and downloaded dynamically at runtime  
- Workflow: User uploads an image → model predicts disease class + confidence + class probabilities  

---

## 🗂 Dataset
- Images of teeth (preprocessing required)  
- 7 distinct classes  
- Dataset balance and class distribution analyzed during visualization  

## ⚙ Preprocessing
- **Normalization:** Standardize pixel values  
- **Augmentation:** Flip, rotate, zoom, and other transformations to improve generalization  
- **Visualization:**  
  - Class distribution (to check dataset balance)  
  - Sample images before and after augmentation  

## 🧠 Model Architecture
- Framework: **TensorFlow** or **PyTorch**  
- Custom CNN designed for teeth classification  
- Provides a **baseline performance** for comparison with pretrained models  
- Baseline serves as foundation for future improvements  

## 📊 Evaluation Metrics
Model performance is assessed using:
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  

## 🚀 Installation & Usage
### 1. Clone the repository
```bash
git clone https://github.com/Muhammed-AlReay/Teeth_Classification_Computer_vision.git
cd Teeth_Classification_Computer_vision

## 📂 Repository Structure

```bash
├── saved/                    # Deployment files
│   ├── app.py                # Streamlit app
│   ├── requirements.txt      # Dependencies
│
├── pretrained_models/        # Model experiments
│   ├── ResNet50.ipynb
│   ├── DenseNet121.ipynb
│   ├── MobileNet.ipynb
│   ├── functional_1.ipynb

