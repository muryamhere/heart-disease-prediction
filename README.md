# Heart Disease Prediction

This project aims to predict heart disease using machine learning techniques. The prediction model is built using a dataset containing various health parameters, and the model is saved for future use.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model](#model)
7. [Results](#results)
8. [License](#license)

## Project Structure

```
Heart Disease Prediction/
├── heart-disease-sample-data.csv
├── new-heart-disease-data.csv
├── heart_disease_model.pkl
├── Heart Disease Prediction.ipynb
└── Heart Disease Prediction.html
```

- **heart-disease-sample-data.csv**: Sample dataset used for training the model.
- **new-heart-disease-data.csv**: New dataset for testing the model.
- **heart_disease_model.pkl**: Serialized machine learning model.
- **Heart Disease Prediction.ipynb**: Jupyter notebook containing the data analysis, model training, and evaluation.
- **Heart Disease Prediction.html**: HTML export of the Jupyter notebook.

## Requirements

- Python 3.7 or above
- Jupyter Notebook
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/muryamhere/heart-disease-prediction
   cd heart-disease-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```

2. Open the `Heart Disease Prediction.ipynb` notebook and run the cells to see the data analysis, model training, and evaluation steps.

3. To use the trained model, you can load it using the following code:
   ```python
   import pickle
   model = pickle.load(open('heart_disease_model.pkl', 'rb'))
   ```

4. Predict using the model:
   ```python
   import pandas as pd
   new_data = pd.read_csv('new-heart-disease-data.csv')
   predictions = model.predict(new_data)
   print(predictions)
   ```

## Data

The datasets used in this project are:
- **heart-disease-sample-data.csv**: Used for training the model.
- **new-heart-disease-data.csv**: Used for testing the model.

Ensure that the datasets are in the same directory as the notebook and the model file.

## Model

The model is built using scikit-learn's machine learning algorithms. The model training and evaluation steps are detailed in the `Heart Disease Prediction.ipynb` notebook.

## Results

The performance of the model is evaluated using various metrics like accuracy, precision, recall, and F1-score. The detailed results can be found in the `Heart Disease Prediction.ipynb` notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
