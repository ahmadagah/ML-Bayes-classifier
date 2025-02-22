# Spam Classification Using Gaussian Naïve Bayes

**Author:** Ahmad Zafar Agah  
**Date:** February 22, 2025

## Description

This project implements a simple Naïve Bayes classifier for spam classification using the Spambase dataset from the UCI Machine Learning Repository. The dataset contains 57 continuous features representing word and character frequencies (including capitalization metrics) and 1 binary label (0 for not spam, 1 for spam). The goal is to train the classifier on a training set and evaluate its performance on a test set.

## Installation

1. **Clone or Download the Repository:**  
   Download the project files to your local machine.

2. **Create a Virtual Environment:**  
   Open a terminal in the project directory and run:
   
   ```bash
   python -m venv venv
   ```


Activate the Virtual Environment:

Windows:
bash
Copy
venv\Scripts\activate
Linux/Mac:
bash
Copy
source venv/bin/activate
Install Required Packages:
A requirements.txt file is provided. Install the necessary packages by running:

bash
Copy
pip install -r requirements.txt
Required packages include: numpy, pandas, seaborn, and matplotlib.

Usage
After installing the requirements, run the main Python script:

bash
Copy
python main.py
This script will load the Spambase dataset, split it into training and test sets (maintaining a 40% spam, 60% not-spam ratio), compute prior probabilities and feature statistics, make predictions using the Gaussian Naïve Bayes classifier, and evaluate the model’s performance (accuracy, precision, recall, and confusion matrix).

Notes
The classifier uses logarithmic probabilities to avoid numerical underflow.
The dataset is split 50%-50% (approximately 2300 instances each) while preserving the original class distribution.
Although Naïve Bayes assumes that features are independent, many features (such as word frequencies) are correlated. Despite this, the model achieves high recall, making it a useful baseline for spam filtering.
