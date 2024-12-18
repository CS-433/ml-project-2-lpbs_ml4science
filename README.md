[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# ML4Science Project @ LPBS Lab

## Project Overview

This project examines the lifespan and behavior of *C. elegans* under two drug treatments and explores their behavior under optogenetic conditions. The primary objectives are to utilize data from the Laboratory of the Physics of Biological Systems (LPBS) to:

1. Classify worms based on drug treatments (**Drug1** and **Drug2**).
2. Predict lifespan-related outcomes.
3. Analyze behavioral responses to light stimuli under optogenetic conditions (**ATR+** and **ATR-**).

For drug classification, a **linear SVM** model performed best considering all 3 classifications, achieving:
- Multi-class classification mean accuracy: **0.544557**
- Binary classification mean accuracies:
  - **Drug1**: **0.741053**
  - **Drug2**: **0.419421**

Models for lifespan estimation demonstrated that **[insert results here]**.

Finally, models analyzing optogenetic behavior revealed that **[insert results here]**.

---

## Acknowledgments

We would like to thank the **Laboratory of the Physics of Biological Systems (LPBS)** for their support, data, and for taking us on as students for this project.

---

## Installation

To install the libraries required for this project, run the following command:

```bash
pip install -r requirements.txt
```

This should install the following libaries

Library | Version | 
--- | --- | 
numpy | 1.21.6 |
pandas | 1.0.0 |
matplotlib | 3.3.1 |
joblib | 1.3.2 |
seaborn | 0.10.0 |
scikit-learn | 1.0.2 |
xgboost | 1.6.2 |

---

## Running the Project

To execute the best-performing models for all three sections (drug classification, lifespan estimation, and optogenetic behavior analysis), run the `run.py` script with the appropriate flag to specify the type of classification:

### Command Structure
```bash
python run.py --type <flag>
```

### Flags
- `classification-drug1`: Runs the model for binary classification of the Drug1 dataset.
- `classification-drug2`: Runs the model for binary classification of the Drug2 dataset.
- `classification-multiclass`: Runs the model for multiclass classification using the combined dataset.

### Example Usage
To run the model for Drug1 classification:
```bash
python run.py --type classification-drug1
```

To run the model for Drug2 classification:
```bash
python run.py --type classification-drug2
```

To run the model for multiclass classification:
```bash
python run.py --type classification-multiclass
```

### Output
- The mean accuracy and standard deviation for the selected model will be printed.
- If a pre-trained model exists in the `models/` directory, it will be loaded. If not, a new model will be trained and saved in the `models/` directory.

The respective datasets are loaded from the `lifespan_merged_datasets/` folder, and the model files are stored in the `models/` folder as:
- `best_model_Drug1.pkl`
- `best_model_Drug2.pkl`
- `best_model_multiclass.pkl`.

## File Structure 

```graphql
ml-project-2-lpbs-ml4science/
│
├── images/                             # Images
│   ├── ... 
│
├── lifespan_merged_datasets/           # Input data files
│   ├── mergedworms_combined.csv        # All 48 worms merged
│   ├── mergedworms_combined2.csv       # All 48 worms merged
│   └── mergedworms_Drug1.csv           # 24 Drug1 and control worms
│   └── mergedworms_Drug2.csv           # 24 Drug2 and control worms
│
├── lifespan_exploration/               # Lifespan-related exploration
│   ├── classification_functions.py     # Classification functions for training model and plotting
│   ├── drugs_classification.ipynb      # Notebook explaining classification
│   └── lifespan_estimation_function.py # Lifespan estimation functions for training model
│   └── lifespan_estimation.ipynb       # Notebook explaining lifespan estimation
│
├── preprocessing/                      # Preprocessing
│   ├── Analysis_single_worm.ipynb      # Lifespan dataset - Initial analysis for feature engineering         
│   ├── lifespan_functions.py           # Lifespan dataset - Preprocessing functions
│   └── lifespan_make_df.ipynb          # Lifespan dataset - Making the dataframe and checking worm death
│
├── models/                             # Saved models
│   ├── best_model_Drug1.pkl            # Drug1 binary classification  
│   ├── best_model_Drug2.pkl            # Drug2 binary classification
│   ├── best_model_multiclass.pkl       # Multi-class classification
│   ├── lifespan_prediction_Drug1.pkl   # Lifespan prediction for Drug1
│   ├── lifespan_prediction_Drug2.pkl   # Lifespan prediction for Drug2
│   ├── lifespan_prediction_all.pkl     # Lifespan prediction for all worms
│   └── best_model_opto.pkl             
│
├── requirements.txt                    # Required libraries
├── run.py                              # Main script to run models
└── README.md                           # Project documentation
```

## Authors

For any questions or clarifications, please reach out to the project authors.

- Advaith Sriram
- Srushti Singh
- Chady Bensaid