# 🦷 Teeth Classification using Deep Learning

## 📌 Project Overview
This project focuses on developing a *comprehensive computer vision solution* for *teeth classification*.  
We experimented with multiple pretrained architectures and a CNN model from scratch, then deployed the best-performing model (ResNet50) using Streamlit for interactive predictions.
The objective is to preprocess and visualize dental images, then build and train a robust deep learning model to classify teeth into *7 distinct categories*.  

Accurate teeth classification plays a vital role in:
- 🏥 Enhancing diagnostic precision  
- 🤖 Supporting AI-driven dental solutions  
- 😀 Improving patient outcomes in healthcare  
---
##🚀 Project Overview
*Goal*: Classify medical images into 7 disease categories:
  CaS, CoS, Gum, MC, OC, OLP, OT
*Models Applied*:
-🧩 ResNet50 ✅ (Best performing, used for deployment)
-🔬 DenseNet121
-📱 MobileNet
-🧠 Vision Transformer (ViT)
-🏗️ CNN from scratch

*Deployment*:
 - Implemented with Streamlit
 - Model weights stored on Google Drive and downloaded dynamically at runtime
 - User uploads an image → model predicts disease class + confidence + class probabilities
---
##📂 Repository Structure

```bash
├── saved/ # Deployment app files
│ ├── app.py # Streamlit app for prediction
│ ├── requirements.txt # Required dependencies
│
├── pretrained_models/ # Pretrained model experiments
│ ├── ResNet50.ipynb
│ ├── DenseNet121.ipynb
│ ├── MobileNet.ipynb
│ ├── functional_1.ipynb
---
---
## 🗂 Dataset
- Images of teeth (preprocessing required).  
- 7 distinct classes for classification.  
- Dataset balance and class distribution to be analyzed during visualization.  
---
---

## ⚙ Preprocessing
- *Normalization*: Ensure consistent pixel value ranges.  
- *Augmentation*: Apply transformations (flip, rotation, zoom, etc.) to improve model generalization.  
- *Visualization*:  
  - Show class distribution (to check dataset balance).  
  - Display images before and after augmentation.  

---

## 🧠 Model Architecture
- Framework: *TensorFlow* or *PyTorch*  
- *Custom CNN model* designed for teeth classification.  
- Trained to establish a *baseline performance*.  
- Baseline will serve as a foundation for future improvements and optimization.  

---

## 📊 Evaluation
The model performance will be assessed using:
- *Accuracy*  
- *Precision*  
- *Recall*  
- *F1-Score*  

---

## 🚀 Installation & Usage
### 1.Clone the repository
```bash
git clone https://github.com/Muhammed-AlReay/Teeth_Classification_Computer_vision
cd Teeth_Classification_Computer_vision
### 2.Create virtual environment (optional but recommended)
```bash
python -m venv venv

source venv/bin/activate     # On Linux/Mac
 
venv\Scripts\activate        # On Windows
### 3.Install dependencies
```bash
 pip install -r requirements.txt
### 4.Run the app
```bash
streamlit run app.py

##🔮Results
#📌 ResNet50 (Best Model) Test Accuracy: 94% Test Loss: 0.2136
ResNet50 achieved the best balance between accuracy and generalization, making it the primary choice for deployment.
Other models (DenseNet121, MobileNet, ViT, CNN from scratch) were also trained and compared for benchmarking.





