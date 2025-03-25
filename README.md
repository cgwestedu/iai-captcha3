# CAPTCHA breaking in Darknet Marketplaces

## Usage
### 1. Object-based CAPTCHA-breaker can be found in `object/`

#### Preprocessing
- `create_training_data.py`: 
  - Converts images into RGBA
  - Permutes data
  - Splits into training & validation
  - Rotates data

    **&rarr; `data/training_data/` & `data/validation_data/`**

#### Training
- `trainer.py`: 
  - Uses `prepare_training.py` to get training/validation images & labels
  - Create one-hot encoding
  - Uses `network.py` to get CNN
  - Training with early stopping & saving best model

    **&rarr; Saved models in `models/`**

#### Testing
- `tester.py`: 
  - **User needs to provide timestamp of model to use**
  - Uses HTML pages in `data/test_data/`
  - Uses `solver.py` to load model & receive prediction
  - Compare prediction & true label
  - Compute accuracy

    **&rarr; Test accuracy**