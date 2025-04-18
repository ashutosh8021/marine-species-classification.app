# 🐟 Marine Species Classifier using Deep Learning and Computer Vision

[![Streamlit](https://img.shields.io/badge/Deployed%20App-Streamlit-success?logo=streamlit)](https://marine-species-classificationapp.streamlit.app/)  
A web-based deep learning application that classifies underwater marine species from images and visualizes model attention using Grad-CAM.

---

## Demo

[Demo](https://drive.google.com/file/d/158qmcR0KzDKsbTYPvKGGQRj4wZO8MT6H/view?usp=sharing)



## 🌐 Live Demo

**▶️ Deployed on Streamlit Cloud:**  
🔗 [marine-species-classificationapp.streamlit.app](https://marine-species-classificationapp.streamlit.app/)

---

## 🧠 About the Project

This project applies deep learning and computer vision to automatically classify different types of marine species from underwater images.

It also includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to interpret what the model is focusing on while making predictions.

---

## 📂 Dataset

**Dataset Source**: [Fish Species Image Dataset (Kaggle)](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

- 💾 ~4,000 cropped images of various marine species
- 📁 Images stored in `/Fish_Data/images/cropped/`
- ✅ Dataset used to train and evaluate a ResNet-based CNN classifier

---

## 🔧 Tech Stack

| Component       | Tool |
|----------------|------|
| Language        | Python 3.12 |
| Deep Learning   | PyTorch, torchvision |
| Web App         | Streamlit |
| Visualization   | Matplotlib, OpenCV |
| Deployment      | Streamlit Cloud |

---

## 🧪 Model Info

- 📚 **Base Model**: ResNet18
- 🔄 Fine-tuned on marine species dataset
- 🧠 Last fully connected layer adjusted for custom class count
- 🧾 Saved in `float16` precision for faster loading

---

## 🚀 Features

- 📤 Upload any marine species image
- 🤖 Get instant top predictions with confidence scores
- 🔍 View Grad-CAM heatmaps to understand model focus
- 📊 Simple, interactive UI

---

## 📁 Project Structure
marine-species-classification.app/
├── app.py                      
├── resnet18_half_precision.pth                  
├── Marine_Species_Classifier_using_Deep_Learning.ipynb                 
├── requirements.txt          
└── README.md                   


👨‍💻 Author
Ashutosh Kumar
🎓 IIT Patna
📧 ashutosh_2312res778@iitp.ac.in
🔗 [LinkedIn](https://www.linkedin.com/in/ashutosh80) | [GitHub](https://github.com/ashutosh8021)

## 🛠️ Local Setup Instructions

```bash
# Clone the repo
git clone https://github.com/ashutosh8021/marine-species-classification.git
cd marine-species-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py



