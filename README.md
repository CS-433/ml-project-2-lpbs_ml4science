[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# ML4Science Project @ LPBS Lab

## Project Overview

This project examines the lifespan and behavior of *C. elegans* under two drug treatments and explores their behavior under optogenetic conditions. The primary objectives are to utilize data from the Laboratory of the Physics of Biological Systems (LPBS) to:

1. Classify worms based on drug treatments (**CompanyDrug** and **Terbinafine**).
2. Predict lifespan-related outcomes.
3. Analyze behavioral responses to light stimuli under optogenetic conditions (**ATR+** and **ATR-**).

For drug classification, a **linear SVM** model performed best, achieving:
- Multi-class classification mean accuracy: **0.544557**
- Binary classification mean accuracies:
  - **CompanyDrug**: **0.741053**
  - **Terbinafine**: **0.419421**

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

--

## Running the Project

To execute the best-performing models for all three sections (drug classification, lifespan estimation, and optogenetic behavior analysis), run the run.py script:

```bash
python run.py
```
Output:
The mean accuracies and standard deviations for the best models in each section will be printed.
The trained models will be saved in the models/ directory.

## File Structure 

```graphql
ml-project-2-lpbs-ml4science/
│
├── lifespan_merged_datasets/           # Input data files
│   ├── mergedworms_combined.csv        # All 48 worms merged
│   ├── mergedworms_combined2.csv       # All 48 worms merged
│   └── mergedworms_companyDrug.csv     # 24 CompanyDrug and control worms
│   └── mergedworms_terbinafin.csv      # 24 Terbinafin and control worms
│
├── preprocessing/                      # Preprocessing
│   ├── Analysis_single_worm.ipynb      # Lifespan dataset - Initial analysis for feature engineering         
│   ├── lifespan_functions.py           # Lifespan dataset - Preprocessing functions
│   └── lifespan_make_df.ipynb          # Lifespan dataset - Making the dataframe and checking worm death
│
├── models/                             # Saved models
│   ├── best_model_companyDrug.pkl      # CompanyDrug binary classification  
│   ├── best_model_terbinafin.pkl       # Terbinafin binary classification
│   ├── best_model_multiclass.pkl       # Multi-class classification
│   ├── best_model_lifespan.pkl         # Lifespan prediction
│   └── best_model_opto.pkl             
│
├── requirements.txt                    # Required libraries
├── run.py                              # Main script to run models
├── exploratory_analysis.ipynb          # Notebook for EDA and preprocessing
├── model_training.ipynb                # Notebook for model training and evaluation
└── README.md                           # Project documentation
```

## Authors

For any questions or clarifications, please reach out to the project authors.

- Advaith Sriram
- Srushti Singh
- Chady Bensaid