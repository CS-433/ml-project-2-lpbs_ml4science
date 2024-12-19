[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# ML4Science Project @ LPBS Lab

## Project Overview

This project examines the lifespan and behavior of *C. elegans* under two drug treatments and explores their behavior under optogenetic conditions. The primary objectives are to utilize data from the Laboratory of the Physics of Biological Systems (LPBS) to:

1. Classify worms based on drug treatments (**Drug1** and **Drug2**).
2. Predict lifespan-related outcomes.
3. Identify and classify the different worms based on their response to the light stimulus?

For drug classification, a **linear SVM** model performed best considering all 3 classifications, achieving:
- Binary classification mean accuracies:
  - **Drug1**: **0.741053**
  - **Drug2**: **0.419421**
- Multi-class classification mean accuracy: **0.544557**

Models for lifespan estimation were not promising, with linear regressions and elastic net regularization showing very high MSEs of 2.62, 7.65, and 7.07 for Drug1, Drug2 and the combined dataset respectively.


Finally, models analyzing optogenetic behavior revealed that an LSTM model gives the best results with an accuracy of 0.9317 and the loss being 0.3659 for the correct identification of worms.

---

## Acknowledgments

We would like to thank the **Laboratory of the Physics of Biological Systems (LPBS)** for their support, data, and for taking us on as students for this project.

---

## Installation

To install the libraries required for this project, run the following command:

```bash
pip install -r requirements.txt
```

The following libraries were used in this project, along with their specific purposes:

| Library        | Version | Purpose                                                                                  |
|----------------|---------|------------------------------------------------------------------------------------------|
| `numpy`        | 1.21.6  | Used for numerical computations, including array manipulations and statistical measures. |
| `pandas`       | 1.0.0   | Used for handling and processing data in the form of DataFrames.                        |
| `matplotlib`   | 3.3.1   | Used for creating visualizations and plots.                                             |
| `seaborn`      | 0.10.0  | Used for enhanced data visualization, such as heatmaps and pair plots.                  |
| `joblib`       | 1.3.2   | Used for saving and loading trained models.                                             |
| `scikit-learn` | 1.0.2   | Used extensively for machine learning, data preprocessing, and evaluation. Key imports include: |
|                |         | - **Metrics**: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `mean_squared_error`, `r2_score` for evaluating model performance. |
|                |         | - **Preprocessing**: `SimpleImputer`, `StandardScaler` for handling missing values and scaling data. |
|                |         | - **Models**: `LinearRegression`, `LogisticRegression`, `RandomForestRegressor`, `ElasticNet`, `SVC`, `KNeighborsClassifier`, `GaussianNB`, `MLPClassifier`, `DecisionTreeClassifier`, and `RandomForestClassifier` for building and testing models. |
|                |         | - **Model Selection**: `GroupKFold` for cross-validation, `StackingClassifier` for combining multiple models. |
| `xgboost`      | 1.6.2   | Used for implementing `XGBRegressor` and `XGBClassifier`, which provide powerful gradient boosting models for regression and classification tasks. |
| `tensorflow`   | 2.18.0    | Used for building and training deep learning models. Key components include:            |
|                |           | - `Sequential`: To define sequential models.                                            |
|                |           | - Layers such as `Conv1D`, `MaxPooling1D`, `Flatten`, `Dense`, `Dropout`, and `LSTM` for creating convolutional and recurrent neural networks. |
|                |           | - Callbacks like `EarlyStopping` and `ReduceLROnPlateau` for optimizing training.        |
|                |           | - Utilities like `to_categorical` for encoding labels for classification tasks.          |
| `keras`        | 3.7.0     | Integrated with TensorFlow to provide easy-to-use APIs for deep learning. Used in conjunction with TensorFlow for creating and training neural networks. |


---

## Running the Project

To execute the best-performing models for all three sections (drug classification, lifespan estimation, and optogenetic behavior analysis), simply run the `run.py` script. The script will automatically train or retrieve the best models for each section, execute them in order (classification, lifespan estimation, and optogenetics), and print the results.

### Command
```bash
python run.py
```

### Output
The `run.py` script will display the following metrics for the best models of each section:

- Drug Classification:
  - Mean accuracy with standard deviation.

- Lifespan Estimation:
  - Mean squared error (MSE) with standard deviation.
  - R-squared with standard deviation.

- Optogenetic Analysis:
  - Test loss.
  - Test accuracy.

### Models and Datasets
The respective datasets are loaded from the `lifespan_merged_datasets/` folder, and the model files are stored in the `models/` folder. There are several pre-trained models available. They are listed below:
- `best_model_Drug1.pkl`            
- `best_model_Drug2.pkl`            
- `best_model_multiclass.pkl`       
- `lifespan_prediction_all.pkl`     
- `lifespan_prediction_Drug1.pkl`   
- `lifespan_prediction_Drug2.pkl`   
- `cnn_model.keras`                 
- `LSTM_model.keras`                  

If you would like to train your own model, please refer to `lifespan_exploration/drugs_classification.ipynb` for classification and `lifespan_exploration/lifespan_estimation.ipynb` for lifespan estimation. 

### Producing the Optogenetics dataset with all the features

To produce the joint dataset for training from the raw csv data files, you may navigate to the `optogen_data` folder and run the following command (insert the path to the csv files as an input): 
```bash
python combineATR.py
```
After you have successfully inserted the path to your raw dataset, the script automatically produces the merged dataset with all the relevant features for you which can later be split into training and testing datasets.  

For more information about the models we have used for training as well as various analyses and experiments we carried out to study the dataset, please refer to `exploration_ATR.ipynb` and `exploration_ATR2.ipynb` in the `optogen_exploration` folder. 

## File Structure 

```graphql
ml-project-2-lpbs-ml4science/
│
├── images/                             # Images
│   ├── ... 
│
├── lifespan_merged_datasets/           # Lifespan - Input data files
│   ├── mergedworms_combined.csv        # All 48 worms merged
│   ├── mergedworms_combined2.csv       # All 48 worms merged
│   └── mergedworms_Drug1.csv           # 24 Drug1 and control worms
│   └── mergedworms_Drug2.csv           # 24 Drug2 and control worms
│
├── lifespan_exploration/                 # Lifespan-related exploration
│   ├── classification_functions.py       # Classification functions for training model and plotting
│   ├── drugs_classification.ipynb        # Notebook explaining classification
│   ├── lifespan_estimation_functions.py  # Lifespan estimation functions for training model
│   └── lifespan_estimation.ipynb         # Notebook explaining lifespan estimation
│
├── preprocessing/                      # Lifespan Dataset- Preprocessing
│   ├── Analysis_single_worm.ipynb      # Initial analysis for feature engineering         
│   ├── lifespan_functions.py           # Preprocessing functions
│   └── lifespan_make_df.ipynb          # Making the dataframe and checking worm death
│
├── optogen_data/                       # Optogenetic Preprocessing and Dataset Creation
│   ├── combineATR.py                   # Merges all raw datasets after preprocessing
│   ├── exploration_ATR.py              # Preprocessing logic
│   └── functionsATR.py                 # Helper functions for preprocessing logic
│
├── optogen_exploration/                # Optogenetic Exploration
│   ├── exploration_ATR.ipynb           # Initial analysis and draft models
│   └── exploration_ATR2.ipynb          # All models trained for optogenetics
│
├── models/                             # Saved models
│   ├── best_model_Drug1.pkl            # Drug1 binary classification  
│   ├── best_model_Drug2.pkl            # Drug2 binary classification
│   ├── best_model_multiclass.pkl       # Multi-class classification
│   ├── lifespan_prediction_Drug1.pkl   # Lifespan prediction for Drug1
│   ├── lifespan_prediction_Drug2.pkl   # Lifespan prediction for Drug2
│   ├── lifespan_prediction_all.pkl     # Lifespan prediction for all worms
│   ├── LSTM_model.keras                # Optogenetics LSTM pre-trained model
│   └── cnn_model.keras                 # Optogenetics CNN pre-trained model
│
├── requirements.txt                    # Required libraries
├── run.py                              # Main script to run and test best models
└── README.md                           # Project documentation
```

## Authors

For any questions or clarifications, please reach out to the project authors.

- Advaith Sriram
- Srushti Singh
- Chady Bensaid
