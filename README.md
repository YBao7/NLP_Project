# Detecting Motivational Regulation Strategies in Student Open-Ended Responses

This repository contains code for a pilot study that uses TF–IDF–based natural language processing (NLP) and machine learning to detect motivational regulation strategies in students’ written reflections on their learning experience. The project focuses on three strategies:

- **Willpower**
- **Performance Self-Talk**
- **Effort**

### Table 1. A condensed coding protocol for motivational regulation strategies: Definitions and examples
![Coding Rubric.png](Coding%20Rubric.png)

Using students’ open-ended responses and human-coded annotations, the project builds interpretable models that approximate human judgments of whether a given strategy is present in a response.

---

## 1. Project Overview

Traditional motivational regulation research often relies on closed-ended questionnaires (e.g., Likert scales). These are efficient but may:

- Cue students toward particular strategies
- Inflate endorsement of some strategies
- Miss nuanced or authentic strategy use that appears only in free text

Open-ended responses solve some of these issues but are **time-consuming to code** at scale.

This project explores whether NLP  can:

1. Identify distinctive lexical patterns in students’ open-ended reflections, and  
2. Use these patterns to predict human-coded motivational regulation strategies.

Concretely, the pipeline:

1. Cleans and preprocesses student responses
2. Builds TF–IDF features
3. Extracts top positive/negative TF–IDF keywords for each strategy
4. Trains **logistic regression models** with leave-one-out cross-validation (LOOCV)**
5. Computes **semantic similarity** between responses and strategy-specific keyword sets using spaCy embeddings

---

## 2. Methods in Brief

### 2.1 Data

- Open-ended responses from middle-school students using the MATHia digital math platform

### Figure 1. Interface of a learning session in the MATHia
![MATHia Demonstration.png](MATHia%20Demonstration.png)

- Students answered this open-ended prompt:  
> “As you were working on this content, how did you motivate yourself? Please describe in as much detail as you can.”

- Human coders annotated each response for whether it shows:
  - **Willpower**
  - **Performance Self-Talk**
  - **Effort**
  - (`0` = strategy absent, `1` = strategy present)

> **Note:** Due to privacy and data-sharing agreements, raw student data are **not** included in this repository. Please adapt the pipeline to your own dataset.

### 2.2 Text Preprocessing

The preprocessing pipeline includes:

- Lowercasing and removal of non-alphabetic characters  
- Detection and removal of **nonsense / gibberish** strings using:
  - `nostril` (nonsense string detector)
  - A trained **gibberish detection** model
- Tokenization and lemmatization using spaCy(`en_core_web_lg`)
- Removal of:
  - Stopwords
  - Non-alphabetic tokens
- Filtering out responses with no remaining valid tokens

The final corpus in the study contained:

- **580** responses  
- **560** unique word features in the TF–IDF matrix

### 2.3 TF–IDF Feature Construction

- Build a bag-of-words representation over lemmatized tokens  
- Compute **TF** (term frequency per response)  
- Compute **IDF** (inverse document frequency across responses)  
- Construct the **TF–IDF matrix** as TF × IDF  
- For each strategy and label (`0` or `1`), compute **average TF–IDF** per word and extract:
  - Top 50 **positive-predictive** words (label = 1)
  - Top 50 **negative-predictive** words (label = 0) 

These keyword sets are used for:

- Feature selection in classification
- Reference texts in semantic similarity analysis

### 2.4 Machine Learning (Logistic Regression + LOOCV)

For each strategy:

1. Create a feature list by combining the top 50 positive and top 50 negative keywords (up to 100 terms)  
2. Subset the TF–IDF matrix to these columns  
3. Handle class imbalance via random undersampling of the majority class  
4. Train a logistic regression classifier with leave-one-out cross-validation (LOOCV) 
5. Compute:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 score  
   - Confusion matrix  

All models achieved performance **well above chance (0.50 accuracy)**, with F1 scores around **0.72–0.79**.

### Figure 2. Confusion matrix heatmaps from machine learning
![ML output.png](ML%20output.png)

### 2.5 Semantic Similarity Analysis

For each strategy:

1. Concatenate the **top 50 positive TF–IDF keywords** into a single “reference document”  
2. Use **spaCy `en_core_web_lg`** to encode:
   - The reference document
   - Each student response
3. Compute **cosine similarity** between each response and the reference document  
4. Compare similarity distributions between:
   - Responses labeled **0** (strategy absent)
   - Responses labeled **1** (strategy present)
5. Use **Welch’s two-sample t-test** to test group differences

Across all three strategies, **strategy-present responses** showed significantly higher similarity to their respective keyword sets than strategy-absent responses.

### Figure 3. Box plots of semantic similarity scores
![Semantic Similarity.png](Semantic%20Similarity.png)
---
