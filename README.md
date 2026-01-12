# Face Recognition System Implementation Report

## 1. Abstract
This project implements a robust real-time Face Recognition system capable of detecting, identifying, and logging individuals via a webcam feed. The system leverages deep learning techniques for face detection and embedding generation, combined with a classical Machine Learning classifier (SVM) for high-accuracy identification. The pipeline supports end-to-end operations from dataset creation to real-time recognition.

## 2. System Architecture

The system operates in a four-stage pipeline:

1.  **Data Acquisition**: Capture and labeling of face images.
2.  **Preprocessing**: Face alignment, validtion, and embedding extraction.
3.  **Model Training**: Training a classifier on 128-dimensional face embeddings.
4.  **Inference**: Real-time detection and recognition.

### 2.1 Technology Stack
-   **face Detection**: Caffe-based Deep Neural Network (SSD framework).
-   **Feature Extraction**: OpenFace (`nn4.small2.v1`) to generate 128-d embeddings.
-   **Classification**: Support Vector Machine (SVM) with Grid Search optimization.
-   **Language/Libraries**: Python 3, OpenCV, Scikit-learn, Imutils.

---

## 3. Methodology & Process Flow

### Step 1: Dataset Creation (`1_datasetCreation.py`)
**Objective**: Build a labeled dataset of user faces.
-   **Mechanism**: The script initializes a video stream and prompts the user for identity details (Name, ID).
-   **Detection**: Uses a pre-trained **ResNet-10 SSD (Single Shot Detector)** model to locate faces. This deep learning approach is superior to Haar Cascades, offering robustness against varying lighting and angles.
-   **Storage**: Valid face ROIs (Regions of Interest) are saved to `dataset/{Name}/` and metadata is logged to `student.csv`.

### Step 2: Preprocessing & Embeddings (`2_preprocessingEmbeddings.py`)
**Objective**: Convert raw images into machine-readable feature vectors.
-   **Filtering**: Images are filtered for quality. Faces smaller than 50x50 pixels or with detection confidence < 60% are discarded to prevent noise.
-   **Embedding**: Each valid face is passed through the **OpenFace** Torch network. This network maps the face into a 128-dimensional hyperspace where similar faces are clustered together (Euclidean distance corresponds to similarity).
-   **Output**: A serialized pickle file (`embeddings.pickle`) containing the vector representations and labels.

### Step 3: Model Training (`3_trainingFaceML.py`)
**Objective**: Train a classifier to distinguish between identities.
-   **Algorithm**: Support Vector Machine (SVM).
-   **Optimization**: A **Grid Search (GridSearchCV)** is performed to automatically tune hyperparameters:
    -   *Kernels*: Linear, RBF, Polynomial.
    -   *C (Regularization)*: Controls trade-off between decision boundary smoothness and classifying training points correctly.
    -   *Gamma*: Defines individual point influence.
-   **Result**: The best performing model is serialized to `recognizer.pickle`.

### Step 4: Real-time Recognition (`4_recognizingPerson.py` / `5_...`)
**Objective**: Identify individuals in a live video stream.
1.  **Frame Capture**: Read frame from webcam.
2.  **Detection**: Locate face using the SSD model.
3.  **Encoding**: Extract 128-d embedding for the detected face.
4.  **Prediction**: Pass embedding to the trained SVM. The model returns the class probability.
5.  **Output**: If probability > Threshold, displays the Name and Confidence overlay. Option 5 additionally fetches metadata (Roll Number) from the CSV.

---

## 4. Performance Improvements & Optimizations

Recent iterations of this system introduced several key optimizations to ensure high accuracy:
1.  **DNN Upgrade**: Replaced Haar Cascades with DNN for data collection, resolving "no face detected" issues.
2.  **Quality Filters**: Implemented strict size and confidence thresholds during preprocessing to eliminate "garbage in, garbage out".
3.  **Hyperparameter Tuning**: Switched from a hardcoded Linear SVM to a dynamic Grid Search, ensuring the model mathematically fits the specific facial features of the users.

## 5. How to Run

A central orchestration script `main.py` is provided.

1.  Install dependencies:
    ```bash
    pip install opencv-python scikit-learn imutils numpy
    ```
2.  Run the system:
    ```bash
    python main.py
    ```
3.  Follow the menu options sequentially (1 -> 2 -> 3 -> 4/5).
