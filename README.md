🧠 Project Overview
This project focuses on high-accuracy image classification for the classic Kaggle "Dogs vs. Cats" challenge. 🐾 The goal is to build a model that can accurately distinguish between images of dogs and cats. It uses a powerful hybrid approach, combining a pre-trained deep learning model for feature extraction with a classic machine learning model for classification.

🔍 Objective
To apply transfer learning by using a pre-trained VGG16 model to extract complex features from the images. These features are then used to train a Support Vector Machine (SVM) classifier, aiming for a high validation accuracy and a successful submission to the Kaggle competition.

### 🧮 Steps Involved

* **Data Preparation**
    Imported the Kaggle "Dogs vs. Cats" dataset.
    Used the `zipfile` library to unzip the `train.zip` and `test1.zip` folders, making the raw `.jpg` images accessible.

* **Feature Extraction (Transfer Learning)**
    Loaded the pre-trained **VGG16** model from Keras/TensorFlow, without its final classification layer (`include_top=False`).
    Used this model as a powerful feature extractor, converting each 224x224 image into a 512-dimension feature vector.

* **Data Preprocessing**
    Selected a sample of 4,000 images for faster training.
    Applied `StandardScaler()` from Scikit-learn to normalize the 512-dimension feature vectors, a critical step for SVM performance.

* **Model Training**
    Trained a **Support Vector Machine (`SVC`)** model from Scikit-learn on the scaled training features and labels (0=Cat, 1=Dog).
    Used a `linear` kernel for high efficiency on this high-dimensional data.

* **Evaluation**
    Used `train_test_split` to create a validation set.
    Evaluated the trained SVM on the unseen validation data, achieving high accuracy (~98-99%) and generating a classification report.

* **Submission Generation**
    Repeated the feature extraction and scaling process for all 12,500 test images.
    Used the trained SVM to predict the probability of "dog" for each test image and saved the results in the `submission.csv` format.

---

### 🛠️ Tools & Libraries Used

* **Python**
* **TensorFlow / Keras** – For loading the VGG16 model and deep feature extraction.
* **Scikit-learn** – For machine learning (`StandardScaler`, `SVC`, `train_test_split`, metrics).
* **Pandas** – For creating the final submission DataFrame.
* **NumPy** – For numerical operations and array manipulation.
* **zipfile** – For unzipping the source dataset.
* **tqdm** – For progress bars during feature extraction.

---

### 📈 Key Takeaways

* Applied **transfer learning** to solve a complex computer vision problem without training a deep learning model from scratch.
* Demonstrated a powerful hybrid approach by combining a pre-trained CNN (VGG16) for feature extraction with a classic ML model (SVM) for classification.
* Strengthened understanding of image preprocessing, feature scaling, and the importance of `StandardScaler` for SVMs.
* Successfully navigated a complete Kaggle competition workflow, from data preparation to final submission.
