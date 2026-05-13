# Processing Code for BRFSS Diabetes Indicator Dataset

This dataset is a cleaned and processed version of the CDC's Behavioral Risk Factor Surveillance System (BRFSS) annual telephone survey data, spanning 20 years from 2005 to 2024. It is designed for diabetes risk prediction, longitudinal health trend analysis, and public health research.

## Features

Each year's dataset contains 33 features grouped into four categories:

- Demographics: age, sex, race, BMI, weight, height, marital status, education level, employment status, income level
- Behavioral: general health, physical/mental/poor health days, exercise, smoking, alcohol consumption
- Clinical: high blood pressure, high cholesterol, stroke, heart attack, coronary heart disease
- Healthcare Access: cost barrier to care, last checkup, personal doctor
- Target: diabetes diagnosis (3 class: Non-diabetic, Pre-diabetic, Diabetic)

Please note that features related to blood pressure, and cholesterol are only available for odd-numbered years (ex. 2005, 2007, etc). The 2 numbers after every variable indicate its version number. Features with the same version number across different years share the same data distribution. So far, only income has had a version increment from 2021 where the number of discrete values went up from 8 to 11. 

`age_00`, `education_00`, `income_00/01`, `gen_health_00`, and `l_checkup_00` are the values from the preceding features, but ordinally encoded to make them easier for modelling. You can use the preceding features for EDA and drop them when modelling. 

This dataset is primarily a multi-class classification problem with high class imbalance. It can be modified to a binary classification problem by merging `Pre-diabetic`, and `Non-diabetic`, and into a balanced class problem by resampling the `Non-diabetic` class till the # of records in that class is similar to the number of records in the `Diabetic` class.

## File Formats

Two formats are provided for each year:

- CSV (csv/) - human readable
- Parquet (parquet/) - smaller file size, preserves dtypes (recommended)

If you wish to load the dataset with the lowest possible memory footprint; load the `parquet` files like this: 

```python
import polars as pl

df = pl.read_parquet("DATASET_20XX.parquet")
```
or like this:

```python
import pandas as pd

df = pd.read_parquet("DATASET_20XX.parquet", dtype_backend="numpy_nullable")
```
For broader compatibility across different libraries and library versions, load the `csv` files like this:

```python
import polars as pl

DTYPE_map = {
    "year": pl.Int64,
    "age_00": pl.Int8,
    "education_00": pl.Int8,
    "income_00": pl.Int8,
    "income_01": pl.Int8,
    "gen_health_00": pl.Int8,
    "physical_health_days_00": pl.Int8,
    "mental_health_days_00": pl.Int8,
    "poor_health_days_00": pl.Int8,
    "avg_drinks_p_day_00": pl.Int8,
    "l_checkup_00": pl.Int8
}

data = pl.read_csv("DATASET_20XY.csv", schema_overrides=DTYPE_MAP)
```
or like this:

```python
import pandas as pd

DTYPE_map = {
    "year": "Int64",
    "age_00": "Int8",
    "education_00": "Int8",
    "income_00": "Int8",
    "income_01": "Int8",
    "gen_health_00": "Int8",
    "physical_health_days_00": "Int8",
    "mental_health_days_00": "Int8",
    "poor_health_days_00": "Int8",
    "avg_drinks_p_day_00": "Int8",
    "l_checkup_00": "Int8"
}

data = pd.read_csv("DATASET_20XY.csv", dtype=DTYPE_map)
```
## Repository Structure
```
main
├─ to_csv.py           # XPT -> CSV conversion script
├─ pyproject.toml      # Dependencies
├─ dataset_features.md # Feature documentation
├─ 20XY
│  └── process_20XY.py # Main processing scripts
└─ README.md
```

## Data Source
- CDC BRFSS Annual Survey Data:
https://www.cdc.gov/brfss/annual_data/annual_data.htm

- BRFSS Diabetes Indicator Dataset on Kaggle: https://www.kaggle.com/datasets/spandanjit2005/brfss-diabetes-indicator-dataset
  

## License

Dataset licensed under [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Processing code licensed under [MIT License](https://www.mit.edu/~amini/LICENSE.md).