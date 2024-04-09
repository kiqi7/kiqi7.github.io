---
layout: post
title:  "Advanced Sentiment Analysis and Multi-classification of Product Comments for Business Intelligence Enhancement"
date:   2021-03-16 05:44:48 +0000
categories: machine-learning
tags: [sentiment-analysis, neural-networks, business-intelligence]
---
## Abstract

In the realm of e-commerce, the ability to analyze customer sentiment accurately presents a significant advantage. This report delineates the development, application, and outcomes of deploying advanced machine learning models, focusing on sentiment analysis of product comments. By processing data from platforms like TIANMAO and incorporating manual labeling, the study showcases the methodologies' efficacy in discerning customer sentiments, which facilitates improved business strategies, customer engagement, and market positioning.

## 1. Introduction

The digitization of consumer feedback in online shopping platforms has resulted in a deluge of unstructured text data, encompassing product reviews, customer service dialogues, and social media interactions. Sentiment analysis stands as a pivotal business intelligence tool, transforming this data into actionable insights. This technique not only reveals consumer perceptions but also uncovers underlying trends and preferences, enabling businesses to tailor their strategies to meet market demands. This report articulates the approach towards applying sentiment analysis to product comments, leveraging a dataset derived from TIANMAO and enhanced through manual annotation.

## 2. Sentiment Analysis: Scope and Implementation

### 2.1 Business Intelligence through Sentiment Analysis

### Advantages

The sentiment analysis model offers multifaceted benefits, including the holistic evaluation of customer interactions across the sales cycle and pinpointing pivotal factors influencing purchasing decisions. Furthermore, it enriches business intelligence by integrating diverse features like conversion rates and customer feedback, culminating in a comprehensive understanding of market dynamics.

### Applications

Sentiment analysis is instrumental in refining business operations, encompassing customer segmentation, product categorization, and enhancing marketing strategies. It provides a granular analysis of consumer sentiment, facilitating targeted marketing and product development.

### 2.2 Methodological Framework

### 2.2.1 Data Pre-Processing

The pre-processing stage is crucial for converting textual data into a format amenable to machine learning models. This involves:

- Tokenization and vectorization of text data.
- Padding sequences to uniform lengths.
- Employing word embeddings to capture semantic relationships between words.

### 2.2.2 Model Building

This segment focuses on constructing a model architecture adept at sentiment classification. The architecture is grounded on Recurrent Neural Networks (RNN), with an emphasis on Long Short-Term Memory (LSTM) networks, renowned for their proficiency in capturing temporal dependencies in text data.

### 2.2.3 Training Phase

The training phase is meticulously designed to mitigate dataset imbalances, enhancing the model's capability to identify negative sentiments accurately. This is pivotal for delineating actionable insights aimed at service improvement.

### 2.2.4 Prediction Dynamics

The prediction phase evaluates the trained model's applicability across diverse product categories, ensuring its robustness and scalability in real-world scenarios.

## 3. Detailed Implementation and Analytical Results

### 3.1 Model Architecture and Hyperparameters

The chosen LSTM model for sentiment analysis on product comments embodies an advanced structure optimized for processing sequential data, reflecting in its adeptness at handling varying lengths of textual input and capturing the nuanced dynamics of language. Below is the detailed breakdown of the model's architecture and hyperparameters, constituting the core of its operational framework:

**Table 4: Hyperparameters for final Model Structure**

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| embedding (Embedding) | (None, 87, 300) | 26,100 |
| bidirectional (Bidirectional) | (None, 87, 512) | 1,140,736 |
| lstm1(LSTM) | (None, 87, 256) | 787,456 |
| lstm2(LSTM) | (None, 128) | 197,120 |
| dropout (Dropout) | (None, 128) | 0 |
| dense (Dense) | (None, 64) | 8,256 |
| dense1(Dense) | (None, 1) | 65 |
| Total params: |  | 2,159,733 |
| Trainable params: |  | 2,133,633 |
| Non-trainable params: |  | 26,100 |

This table articulates the sequential build of the LSTM model, starting from an embedding layer that maps words to a dense vector of fixed size, followed by a bidirectional layer that allows the network to have insights from both past (backward) and future (forward) states. Subsequent LSTM layers enhance the model's ability to capture long-term dependencies, complemented by a dropout layer to prevent overfitting. The dense layers towards the end serve to consolidate the learned features into a final output that predicts the sentiment of the input text.

The architecture underscores the LSTM model's capability to intricately process and analyze textual data, making it an invaluable asset in sentiment analysis applications. The careful calibration of parameters, such as the number of units in each layer and the dimensionality of the embeddings, has been instrumental in optimizing the model for high accuracy and performance in classifying sentiments from product comments.

## 3.2 Results Analysis

Following the implementation of the LSTM model, as detailed in Table 4, the subsequent phases of training and testing revealed a high degree of accuracy in sentiment classification. This section elaborates on the model's performance, emphasizing its robustness and effectiveness in real-world applications.

### 3.3 Hardware and Software Requirements

The deep learning environment necessitates specific hardware and software configurations to optimize model training and execution, detailed in **Table 8**.

**Table 8: Hardware and Software Requirements List**

| Hardware/Software | Specification |
| --- | --- |
| Graphics Processing Unit (GPU) | NVIDIA GeForce RTX 2060 SUPER |
| Jupyter Notebook   | 6.0.3        |
| Python | 3.8.0 |
| CUDA | V10.0  |
| cuDNN | cuDNN64 7.dll  |

**Table 9: Python Libraries Requirements List**

| Library Name | Version |
| --- | --- |
| TensorFlow | 2.3.0 |
| Keras | 2.4.0 |
| Numpy | 1.18.5 |
| pandas | 1.0.5 |
| sklearn | 0.23.2 |
| gensim | 3.8.3 |

### 3.4 Evaluation and Results

### Training Procedure

The training procedure emphasized model performance evaluation through various metrics, with the results showcasing the model's high accuracy and precision in sentiment detection. The development and test accuracies, alongside the True Negative (TN) ratios, are summarized in **Table 10**.

**Table 10: Development Results for Various Configurations of Stack LSTM Models**

| Model | Model Accuracy | Test Accuracy | TN/(TN + FN) | TN/TN + FP |
| --- | --- | --- | --- | --- |
| Model 1 | 85% | 79.54% | 88.27% | 73.9% |
| Model 2 | 79.36% | 79.81% | 81.55% | 84.83% |
| Model 3 | 86.01% | 86.32% | 96.6% | 64.03% |
| Model 4 | 91.31% | 90.80% | 97.48% | 75.81% |

### Classification Results

The classification results detail the model's capability to discern between different sentiments across two datasets, as seen in the performance metrics in **Tables 11 and 12**.

**Table 11: DataSet Count 18037 Test Results**

| Score Name | Results |
| --- | --- |
| Training Data Count | 12625 |
| Test Data Count | 5412 |
| Training Epoch | 10 |
| Batch Size | 128 |
| Model Accuracy | 93.18% |
| Test Accuracy | 88.51% |
| TN/(TN + FN) | 91.34% |
| TN/RN | 95.36% |

**Table 12: Hardware and Software Requirements List** is a repeat of the previously mentioned Table 8, hence not reiterated for brevity.

## 4. Conclusions and Future Work

The deployment of LSTM models in sentiment analysis of product comments has demonstrated significant promise, offering deep insights into consumer behavior and sentiment. This report confirms the viability of using advanced AI techniques to enhance business intelligence, enabling targeted marketing strategies and product development based on consumer feedback.

### Future Directions

The report suggests the integration of attention mechanisms to improve model responsiveness to the varying importance of words in sentiment analysis. This advancement aims to address the current limitations of LSTM models, including their tendency to overlook the context and significance of specific terms in lengthy sentences.

Furthermore, the exploration of transfer learning and the implementation of more sophisticated neural network architectures, such as Transformer models, could provide further enhancements in sentiment analysis accuracy and efficiency.

## Acknowledgments

Gratitude is extended to the data annotation and business team for their meticulous efforts in preparing the business dataset and to the technical team for their dedication in model deploy.