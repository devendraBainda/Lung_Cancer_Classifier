# Lung Cancer Classification System

This is an AI-powered web application for lung cancer tissue classification. This system uses a fine-tuned ResNet50 deep learning model to classify lung tissue images into three categories: benign tissue, squamous cell carcinoma, and adenocarcinoma.

![LungScanAI Demo](https://i.imgur.com/bI0c9vH.png)

## Features

- **AI-Powered Analysis**: Utilizes a deep learning model to classify lung tissue images
- **User-Friendly Interface**: Drag and drop interface for easy image upload
- **Instant Results**: Get classification results with confidence scores in seconds
- **Batch Processing**: Analyze multiple images simultaneously
- **Detailed Reports**: View comprehensive analysis results with visualizations
- **Responsive Design**: Works on desktop and mobile devices

## Dataset

This project was trained on the Lung and Colon Cancer Histopathological Images dataset from Kaggle:
[https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

The dataset includes:
- Lung Benign Tissue
- Lung Squamous Cell Carcinoma
- Lung Adenocarcinoma

## Model Architecture

The classification model is based on ResNet50 with transfer learning:
- Pre-trained ResNet50 base (trained on ImageNet)
- Global Average Pooling layer
- Dense layer with 256 neurons and ReLU activation
- Output layer with softmax activation for 3-class classification

The model achieves over 98% accuracy on the validation dataset.

## Installation

1. Clone this repository with Git LFS to properly download the model file:
```bash
# Make sure Git LFS is installed first
git lfs install

# Clone the repository
git clone https://github.com/devendraBainda/lung_cancer_classifier.git
cd lung_cancer_classifier
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Upload lung tissue images for analysis through the web interface

4. View the classification results and analysis report

## Project Structure

```
lungscanai/
├── app.py                  # Flask application
├── Training_model.py       # Model training script
├── Lung_cancer_prediction.keras  # Pre-trained model (stored with Git LFS)
├── static/
│   ├── css/
│   │   └── styles.css      # Application styles
│   ├── js/
│   │   ├── script.js       # Main JavaScript
│   │   └── results.js      # Results page JavaScript
│   └── uploads/            # Folder for uploaded images
├── templates/
│   ├── index.html          # Main page template
│   └── results.html        # Results page template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## For Developers

If you want to retrain the model on your own dataset:

1. Organize your dataset in the following structure:
```
lung_image_set/
├── Lung_benign_tissue/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Lung_squamous_cell_carcinoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Lung_adenocarcinoma/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. Run the training script:
```bash
python Training_model.py
```

3. The script will save the trained model as `Lung_cancer_prediction.keras`

## Important Note

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or suggestions, please open an issue in this repository.
