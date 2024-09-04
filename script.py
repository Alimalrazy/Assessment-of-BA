import os
import sys
import pdfplumber
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Check if directory path is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

# Get the directory path from command-line argument
directory_path = sys.argv[1]

# Convert the path to a platform-independent format
directory_path = Path(directory_path).resolve()

# Set paths dynamically based on the provided directory
dataset_path = 'data\data\data'
save_path = directory_path / 'ModelFiles'
output_csv = directory_path / 'categorized_resumes.csv'

# Ensure the directory to save models exists
save_path.mkdir(parents=True, exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Function to clean and tokenize extracted text
def clean_and_tokenize_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip leading/trailing spaces

    tokens = nltk.word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Join tokens back into a string

# Function to prepare the training data from the dataset
def prepare_training_data(dataset_path):
    file_names, texts, labels = [], [], []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                cleaned_text = clean_and_tokenize_text(text)
                texts.append(cleaned_text)

                category = os.path.basename(root)  # Use directory name as category
                labels.append(category)
                file_names.append(os.path.splitext(file)[0])  # File name without extension
    return file_names, texts, labels

# Function to examine the dataset and understand the distribution of categories
def examine_dataset_distribution(labels):
    category_distribution = pd.Series(labels).value_counts()
    print("Category Distribution:")
    print(category_distribution)

    # Plot the distribution
    category_distribution.plot(kind='bar', color='skyblue')
    plt.title('Category Distribution in Dataset')
    plt.xlabel('Category')
    plt.ylabel('Number of Resumes')
    plt.xticks(rotation=45, ha="right")
    plt.show()

# Prepare the training data
file_names, texts, labels = prepare_training_data(dataset_path)

# Examine the dataset to understand the distribution of categories
examine_dataset_distribution(labels)

# Split the data into training, validation, and test sets (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp, file_train, file_temp = train_test_split(
    texts, labels, file_names, test_size=0.4, random_state=42, stratify=labels)

X_val, X_test, y_val, y_test, file_val, file_test = train_test_split(
    X_temp, y_temp, file_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Vectorize the text using only the training data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
X_val_vectorized = tfidf_vectorizer.transform(X_val)
X_test_vectorized = tfidf_vectorizer.transform(X_test)

# Encode the labels using only the training data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_vectorized, y_train_encoded)

# Save the models and vectorizer
joblib.dump(tfidf_vectorizer, save_path / 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, save_path / 'label_encoder.pkl')
joblib.dump(svm_model, save_path / 'svm_resume_classifier.pkl')

# Function to predict category of the text and return the actual label
def predict_category(text, vectorizer, model, label_encoder):
    text_vector = vectorizer.transform([text])  # Vectorize the cleaned text
    predicted_label = model.predict(text_vector)[0]  # Get the numeric prediction
    return label_encoder.inverse_transform([predicted_label])[0]  # Convert to actual label

# Function to evaluate the model's performance on a given set
def evaluate_model(svm_model, X, y_true, label_encoder, set_name="Validation"):
    y_pred = svm_model.predict(X)

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{set_name} Set Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Display the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"{set_name} Set Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Evaluate the model on the validation and test sets
evaluate_model(svm_model, X_val_vectorized, y_val_encoded, label_encoder, set_name="Validation")
evaluate_model(svm_model, X_test_vectorized, y_test_encoded, label_encoder, set_name="Test")

# Function to process each PDF in the test set and generate the required CSV data
def process_test_pdfs_to_csv(file_names, test_texts, output_file):
    data = []

    for file_name, text in zip(file_names, test_texts):
        cleaned_text = clean_and_tokenize_text(text)
        category = predict_category(cleaned_text, tfidf_vectorizer, svm_model, label_encoder)

        # Append data to the list with 2 specific columns: filename and Category
        data.append([file_name, category])

    # Create a DataFrame with the specific columns
    df = pd.DataFrame(data, columns=['filename', 'category'])
    df.to_csv(output_file, index=False)
    print(f"CSV file saved to: {output_file}")

# Process the test set and save the data to a CSV file
process_test_pdfs_to_csv(file_test, X_test, output_csv)

# Verify the saved files
print(f"TF-IDF Vectorizer saved to: {save_path / 'tfidf_vectorizer.pkl'}")
print(f"Label Encoder saved to: {save_path / 'label_encoder.pkl'}")
print(f"SVM Model saved to: {save_path / 'svm_resume_classifier.pkl'}")
