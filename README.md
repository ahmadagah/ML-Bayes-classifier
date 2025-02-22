# Spam Classification Using Gaussian Naïve Bayes

**Author:** Ahmad Zafar Agah  
**Date:** February 22, 2025

## Description

This project implements a simple Naïve Bayes classifier for spam classification using the Spambase dataset from the UCI Machine Learning Repository. The dataset contains 57 continuous features representing word and character frequencies (including capitalization metrics) and 1 binary label (0 for not spam, 1 for spam). The goal is to train the classifier on a training set and evaluate its performance on a test set.

## Installation

1. **Clone the Repository:**  
   Open a terminal and run:

   ```bash
   git clone https://github.com/ahmadagah/ML-Bayes-classifier.git
   cd ML-Bayes-classifier
   ```

2. **Create a Virtual Environment:**  
   In the project directory, run:

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**  
   - **Windows:**  

     ```bash
     venv\Scripts\activate
     ```

   - **Linux/Mac:**  

     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages:**  
   Install the necessary packages using:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the requirements, run the main Python script:

```bash
python main.py
```

This script loads the Spambase dataset, splits it into training and test sets (maintaining a 40% spam, 60% not-spam ratio), computes prior probabilities and feature statistics, makes predictions using the Gaussian Naïve Bayes classifier, and evaluates the model’s performance (accuracy, precision, recall, and confusion matrix).

## Notes

- The classifier uses logarithmic probabilities to avoid numerical underflow.
- The dataset is split 50%-50% (approximately 2300 instances each) while preserving the original class distribution.
- Although Naïve Bayes assumes that features are independent, many features (e.g., word frequencies) are correlated. Despite this, the model achieves high recall, making it a useful baseline for spam filtering.
