# Author: Ahmad Zafar Agah
# Date: February 22 2025
# Description: This is a simple implementation of Naive Bayes algorithm for spam classification.
# The dataset used is the Spambase dataset from UCI repository. The dataset contains 57 features
# and 1 label. The features are continuous and represent the frequency of certain words and characters
# in emails. The label is binary (0 or 1) and represents whether an email is spam or not. The goal is to
# train a Naive Bayes classifier on the training set and evaluate its performance on the test set.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
column_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total",
    "label"
]

data = pd.read_csv(url, header=None, names=column_names)

# Check dataset info
# print(data.info())
# print(data.head())
# print(data['label'].value_counts())


# Ensure reproducibility
np.random.seed(42)

# Shuffle indices
shuffled_indices = np.random.permutation(len(data))

# 50% Train - 50% Test
train_size = int(0.5 * len(data))
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

# Split features and labels
X_train, X_test = data.iloc[train_indices, :-1].values, data.iloc[test_indices, :-1].values
y_train, y_test = data.iloc[train_indices, -1].values, data.iloc[test_indices, -1].values

# Print class distribution to ensure 40% spam, 60% not-spam
print("Training set class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))

print("Test set class distribution:")
print(pd.Series(y_test).value_counts(normalize=True))

# Compute prior probabilities
P_spam = np.mean(y_train)  # P(1)
P_not_spam = 1 - P_spam    # P(0)

print(f"P(spam) = {P_spam:.4f}")
print(f"P(not spam) = {P_not_spam:.4f}")


# Compute mean and standard deviation for each feature given spam and not spam
mean_spam = np.mean(X_train[y_train == 1], axis=0)
std_spam = np.std(X_train[y_train == 1], axis=0)
mean_not_spam = np.mean(X_train[y_train == 0], axis=0)
std_not_spam = np.std(X_train[y_train == 0], axis=0)

# Replace zero standard deviations with a small value to avoid divide-by-zero errors
std_spam = np.where(std_spam == 0, 0.0001, std_spam)
std_not_spam = np.where(std_not_spam == 0, 0.0001, std_not_spam)

# Print some values for verification
print("Mean and Standard Deviation for a few features (Spam Class):")
for i in range(5):  # Print for first 5 features
    print(f"Feature {i}: mean={mean_spam[i]:.4f}, std={std_spam[i]:.4f}")

def gaussian_pdf(x, mean, std):
    """Computes the Gaussian probability density function."""
    std = np.where(std == 0, 0.0001, std)  # Ensure non-zero std
    pdf = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    # Prevent zero probabilities (replace zeros with a very small value)
    pdf = np.where(pdf == 0, 1e-10, pdf)
    
    return pdf

def predict(X):
    """Predict class labels using Naïve Bayes."""
    # Compute log probabilities with a small value added
    spam_probs = np.log(P_spam) + np.sum(np.log(gaussian_pdf(X, mean_spam, std_spam) + 1e-10), axis=1)
    not_spam_probs = np.log(P_not_spam) + np.sum(np.log(gaussian_pdf(X, mean_not_spam, std_not_spam) + 1e-10), axis=1)

    # Predict class with higher probability
    return (spam_probs > not_spam_probs).astype(int)



# Make predictions on the test set
y_pred = predict(X_test)


# Compute evaluation metrics
accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)

# Compute confusion matrix
conf_matrix = np.array([
    [np.sum((y_pred == 0) & (y_test == 0)), np.sum((y_pred == 1) & (y_test == 0))],
    [np.sum((y_pred == 0) & (y_test == 1)), np.sum((y_pred == 1) & (y_test == 1))]
])

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)



plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Naïve Bayes Spam Classification")
plt.show()




