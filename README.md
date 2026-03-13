# XAI Analysis of ResNet-34 Classification

This repository contains the code developed as part of the engineering thesis created by *Olimpia* (me) and **Łukasz Cieślik (GitHub: TreF0)**.

**Thesis title:**

ANALYSIS AND VISUALIZATION OF THE CLASSIFICATION PROCESS IN RESNET-34 USING XAI METHODS

The repository contains multiple scripts used for experiments, analysis, and visualization performed during the thesis.

The work focuses on analyzing the decision-making process of a **ResNet-34 neural network** using different **Explainable Artificial Intelligence (XAI)** techniques.

---

# Datasets

The experiments were conducted on two datasets:

- **Swedish Leaf Dataset** – leaf classification
- **HAM10000** – skin lesion classification

For both datasets the neural network was trained and later analyzed using several XAI methods.

---

# Implemented XAI Methods

The following explainability techniques were implemented and analyzed:

- **Layer-wise Relevance Propagation (LRP)**
- **DeepSHAP**
- **Grad-CAM**
- **Score-CAM**
- **Layer-CAM**

---

# Project Goal

The main objective of the project was to **analyze and visualize how the ResNet-34 model makes classification decisions**.

The workflow included:

1. Training the ResNet-34 model on both datasets  
2. Generating explanation maps using multiple XAI techniques  
3. Performing statistical analysis of the explanations  
4. Comparing the behavior of different XAI methods

---

# Additional Information

This repository contains many scripts used for analysis, statistics, normalization, entropy calculations, and comparison of different explanation methods.

For a detailed description of the methodology, experiments, and conclusions, please refer to the **thesis document included in this repository**.

---

# Result

Thanks to this project and thesis, we successfully completed our engineering degree and received the grade **very good**.

And with that… we officially became engineers! 😊
