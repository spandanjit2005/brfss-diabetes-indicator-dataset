import os
import time
import polars as pl

START_2024 = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_csv_path = os.path.join(script_dir, "2024_BRFSS_RAW.csv")

processed_csv_path = os.path.join(script_dir, "DATASET_2024.csv")
processed_parquet_path = os.path.join(script_dir, "DATASET_2024.parquet")

data_2024 = pl.read_csv(raw_csv_path, ignore_errors=True)

# Extract and rename relevant features
data_2024 = data_2024.select([
    # Demographics
    "_AGEG5YR", "SEXVAR", "_RACE",
    "MARITAL", "EDUCA", "EMPLOY1", "INCOME3", "WEIGHT2", "HEIGHT3",

    # Behavioral
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "EXERANY2", "SMOKE100", "DRNKANY6", "AVEDRNK4",

    # Clinical
    "CVDSTRK3", "CVDINFR4", "CVDCRHD4",

    # Healthcare Access
    "MEDCOST1", "CHECKUP1", "PERSDOC3",

    # Target
    "DIABETE4"
])

NEW_FEATURE_NAME_map = {
    # Demographics
    "_AGEG5YR": "age_group_00", "SEXVAR": "sex_00", "_RACE": "race_00",
    "WEIGHT2": "weight_00", "HEIGHT3": "height_00", "MARITAL": "marital_status_00",
    "EDUCA": "education_level_00", "EMPLOY1": "employment_status_00", "INCOME3": "income_level_01",

    # Behavioral
    "GENHLTH": "general_health_00", "PHYSHLTH": "physical_health_days_00", "MENTHLTH": "mental_health_days_00",
    "POORHLTH": "poor_health_days_00", "EXERANY2": "exercise_00", "SMOKE100": "smoked_100_cigarettes_00",
    "DRNKANY6": "drinks_alcohol_00", "AVEDRNK4": "avg_drinks_p_day_00",

    # Clinical
    "CVDSTRK3": "had_stroke_00", "CVDINFR4": "had_heart_attack_00", "CVDCRHD4": "had_coronary_heart_disease_00",

    # Healthcare Access
    "MEDCOST1": "cost_barrier_00", "CHECKUP1": "last_checkup_00", "PERSDOC3": "has_personal_doctor_00",

    # Target
    "DIABETE4": "diabetes"
}

data_2024 = data_2024.rename(NEW_FEATURE_NAME_map)

# Cast columns into specific data types & define `BINARY_map`
INT_8_cols = [
    "age_group_00", "sex_00", "race_00", 
    "marital_status_00", "education_level_00", "employment_status_00", 

    "general_health_00", "physical_health_days_00", "mental_health_days_00", 
    "exercise_00", 

    "had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00",

    "cost_barrier_00", "last_checkup_00", "has_personal_doctor_00",

    "diabetes"
]

data_2024 = data_2024.with_columns([
    pl.col(col).cast(pl.Int8, strict=False)
    for col in INT_8_cols
])

FLOAT_64_cols = [
    "weight_00", "height_00", "income_level_01",
    "poor_health_days_00", "smoked_100_cigarettes_00", "drinks_alcohol_00", "avg_drinks_p_day_00"
]

data_2024 = data_2024.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in FLOAT_64_cols
])

BINARY_map = {
    1: True,
    2: False
}

# Add `year` = 2024
data_2024 = data_2024.with_columns(pl.lit(2024).cast(pl.Int64).alias("year"))

# Process `age_group_00`, & create `age_00` for modelling
AGE_GROUP_map = {
    1: "18 to 24",
    2: "25 to 29",
    3: "30 to 34",
    4: "35 to 39",
    5: "40 to 44",
    6: "45 to 49",
    7: "50 to 54",
    8: "55 to 59",
    9: "60 to 64",
    10: "65 to 69",
    11: "70 to 74",
    12: "75 to 79",
    13: "80 or older"
}

data_2024 = data_2024.with_columns(
    pl.when((pl.col("age_group_00") >= 1) & (pl.col("age_group_00") <= 13)).then(pl.col("age_group_00"))
    .otherwise(None)
    .alias("age_00")
)

data_2024 = data_2024.with_columns(
    pl.col("age_group_00").replace_strict(AGE_GROUP_map, default=None)
    .cast(pl.Categorical)
    .alias("age_group_00")
)

# Process `sex_00`
data_2024 = data_2024.with_columns(
    pl.when(pl.col("sex_00") == 1).then(pl.lit("Male"))
    .when(pl.col("sex_00") == 2).then(pl.lit("Female"))
    
    .otherwise(None)
    .cast(pl.Categorical)
    .alias("sex_00")
)

# Process `race_00`
RACE_map = {
    1: "White",
    2: "Black",
    3: "Asian",
    4: "Native Hawaiian or other Pacific Islander",
    5: "American Indian or Alaskan Native",
    6: "Other",
    7: "Multiracial",
    8: "Hispanic"
}

data_2024 = data_2024.with_columns(
    pl.col("race_00").replace_strict(RACE_map, default=None)
    .cast(pl.Categorical)
    .alias("race_00")
)

# Process `weight_00`
data_2024 = data_2024.with_columns(
    pl.when((pl.col("weight_00") >= 50) & (pl.col("weight_00") <= 999))
    .then(pl.col("weight_00"))

    .when((pl.col("weight_00") >= 9000) & (pl.col("weight_00") <= 9998))
    .then(((pl.col("weight_00") - 9000) * 2.20462).round(2))

    .otherwise(None)
    .cast(pl.Float64)
    .alias("weight_00")
)

# Process `height_00`
data_2024 = data_2024.with_columns(
    pl.when((pl.col("height_00") >= 200) & (pl.col("height_00") <= 711))
    .then((pl.col("height_00") // 100) * 12 + (pl.col("height_00") % 100))

    .when((pl.col("height_00") >= 9000) & (pl.col("height_00") <= 9998))
    .then(((pl.col("height_00") - 9000) / 2.54).round(2))

    .otherwise(None)
    .cast(pl.Float64)
    .alias("height_00")
)

# Compute `bmi`
data_2024 = data_2024.with_columns(
    (pl.col("weight_00") / (pl.col("height_00") ** 2) * 703).round(2)
    .cast(pl.Float64)
    .alias("bmi_00")
)

# Process `marital_status_00`
MARITAL_STATUS_map = {
    1: "Married",
    2: "Divorced",
    3: "Widowed",
    4: "Separated",
    5: "Never married",
    6: "A member of an unmarried couple"
}

data_2024 = data_2024.with_columns(
    pl.col("marital_status_00").replace_strict(MARITAL_STATUS_map, default=None)
    .cast(pl.Categorical)
    .alias("marital_status_00")
)

# Process `education_level_00`, & create `education_00` for modelling
EDUCATION_LEVEL_map = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1 - 8 (Elementary)",
    3: "Grades 9 - 11 (Some high school)",
    4: "Grade 12 or GED (High school graduate)",
    5: "College 1 year to 3 years (Some college or technical school)",
    6: "College 4 years or more (College graduate)"
}

data_2024 = data_2024.with_columns(
    pl.when((pl.col("education_level_00") >= 1) & (pl.col("education_level_00") <= 6))
    .then(pl.col("education_level_00"))

    .otherwise(None)
    .alias("education_00")
)

data_2024 = data_2024.with_columns(
    pl.col("education_level_00").replace_strict(EDUCATION_LEVEL_map, default=None)
    .cast(pl.Categorical)
    .alias("education_level_00")
)

# Process `employment_status_00`
EMPLOYMENT_STATUS_map = {
    1: "Employed for wages",
    2: "Self-employed",
    3: "Out of work for more than 1 year",
    4: "Out of work for less that 1 year",
    5: "A homemaker",
    6: "A student",
    7: "Retired",
    8: "Unable to work"
}

data_2024 = data_2024.with_columns(
    pl.col("employment_status_00").replace_strict(EMPLOYMENT_STATUS_map, default=None)
    .cast(pl.Categorical)
    .alias("employment_status_00")
)

# Process `income_level_01`, & create `income_01` for modelling
INCOME_LEVEL_map = {
    1: "Less than $10,000",
    2: "$10,000 to less than $15,000",
    3: "$15,000 to less than $20,000",
    4: "$20,000 to less than $25,000",
    5: "$25,000 to less than $35,000",
    6: "$35,000 to less than $50,000",
    7: "$50,000 to less than $75,000",
    8: "$75,000 to less than $100,000",
    9: "$100,000 to less than $150,000",
    10: "$150,000 to less than $200,000",
    11: "$200,000 or more"
}

data_2024 = data_2024.with_columns(
    pl.when((pl.col("income_level_01") >= 1) & (pl.col("income_level_01") <= 11))
    .then(pl.col("income_level_01"))

    .otherwise(None)
    .cast(pl.Int8)
    .alias("income_01")
)

data_2024 = data_2024.with_columns(
    pl.col("income_level_01").replace_strict(INCOME_LEVEL_map, default=None)
    .cast(pl.Categorical)
    .alias("income_level_01")
)

# Process `general_health_00`, & create `gen_health_00` for modelling
GENERAL_HEALTH_map = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

data_2024 = data_2024.with_columns(
    pl.when((pl.col("general_health_00") >= 1) & (pl.col("general_health_00") <= 5))
    .then(pl.col("general_health_00"))

    .otherwise(None)
    .alias("gen_health_00")
)

data_2024 = data_2024.with_columns(
    pl.col("general_health_00").replace_strict(GENERAL_HEALTH_map, default=None)
    .cast(pl.Categorical)
    .alias("general_health_00")
)

# Process `physical_health_days_00`, `mental_health_days_00`, & `poor_health_days_00`
def process_health_days(col: str) -> pl.Expr:

    return (
        pl.when((pl.col(col) >= 1) & (pl.col(col) <= 30)).then(pl.col(col))
        
        .when(pl.col(col) == 88).then(pl.lit(0))

        .otherwise(None)
        .cast(pl.Int8)
        .alias(col)
    )

HEALTH_DAY_cols = ["physical_health_days_00", "mental_health_days_00", "poor_health_days_00"]

data_2024 = data_2024.with_columns(
    [process_health_days(col) for col in HEALTH_DAY_cols]
)

# Process `exercise_00`, `smoked_100_cigarettes_00`, & `drinks_alcohol_00`
BINARY_BEHAVIORAL_cols = ["exercise_00", "smoked_100_cigarettes_00", "drinks_alcohol_00"]

data_2024 = data_2024.with_columns([
    pl.col(col).replace_strict(BINARY_map, default=None)
    .cast(pl.Boolean)
    .alias(col)
    for col in BINARY_BEHAVIORAL_cols
])

# Process `avg_drinks_p_day_00`
data_2024 = data_2024.with_columns(
    pl.when(((pl.col("avg_drinks_p_day_00") >= 1) & (pl.col("avg_drinks_p_day_00") <= 76)))
    .then(pl.col("avg_drinks_p_day_00"))

    .when(pl.col("avg_drinks_p_day_00") == 88).then(pl.lit(0))

    .otherwise(None)
    .cast(pl.Int8)
    .alias("avg_drinks_p_day_00")
)

# Process `had_stroke_00`, `had_heart_attack_00`, & had_coronary_heart_disease_00
BINARY_CLINICAL_cols = ["had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00"]

data_2024 = data_2024.with_columns([
    pl.col(col).replace_strict(BINARY_map, default=None)
    .cast(pl.Boolean)
    .alias(col)
    for col in BINARY_CLINICAL_cols
])

# Process `cost_barrier_00`
data_2024 = data_2024.with_columns([
    pl.col("cost_barrier_00").replace_strict(BINARY_map, default=None)
    .cast(pl.Boolean)
    .alias("cost_barrier_00")
])

# Process `last_checkup_00`, & create `l_checkup_00` for modelling
CHECKUP_map = {
    1: "Within the past year (< 12 months ago)",
    2: "Within the past 2 years (1 to <2 years ago)",
    3: "Within the past 5 years (2 to <5 years ago)",
    4: "5 or more years ago",
    8: "Never"
}

data_2024 = data_2024.with_columns(
    pl.when((pl.col("last_checkup_00") >= 1) & (pl.col("last_checkup_00") <= 4))
    .then(pl.col("last_checkup_00"))

    .when(pl.col("last_checkup_00") == 8).then(pl.lit(5))

    .otherwise(None)
    .alias("l_checkup_00")
)

data_2024 = data_2024.with_columns([
    pl.col("last_checkup_00").replace_strict(CHECKUP_map, default=None)
    .cast(pl.Categorical)
    .alias("last_checkup_00")
])

# Process `has_personal_doctor_00`
PERSONAL_DOCTOR_map = {
    1: "Yes, only one",
    2: "More than one",
    3: "No"
}

data_2024 = data_2024.with_columns([
    pl.col("has_personal_doctor_00").replace_strict(PERSONAL_DOCTOR_map, default=None)
    .cast(pl.Categorical)
    .alias("has_personal_doctor_00")
])

# Process `diabetes`
DIABETES_map = {
    1: "Diabetic",
    2: "No",
    3: "No",
    4: "Pre-diabetic"
}

data_2024 = data_2024.with_columns([
    pl.col("diabetes").replace_strict(DIABETES_map, default=None)
    .cast(pl.Categorical)
    .alias("diabetes")
])

# Drop rows with no usable data for the feature column
data_2024 = data_2024.filter(pl.col("diabetes").is_not_null())

# Drop duplicates across all columns
data_2024 = data_2024.unique(keep="first")

# Order the features appropriately
FEATURE_ORDER = [
    # Demographics
    "year", "age_group_00", "age_00", "sex_00", "race_00",
    "weight_00", "height_00", "bmi_00", "marital_status_00",
    "education_level_00", "education_00", "employment_status_00", "income_level_01", "income_01",

    # Behavioral
    "general_health_00", "gen_health_00", "physical_health_days_00", "mental_health_days_00",
    "poor_health_days_00", "exercise_00", "smoked_100_cigarettes_00",
    "drinks_alcohol_00", "avg_drinks_p_day_00",

    # Clinical
    "had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00",

    # Healthcare Access
    "cost_barrier_00", "last_checkup_00", "l_checkup_00", "has_personal_doctor_00",

    # Target
    "diabetes"
]

data_2024 = data_2024.select(FEATURE_ORDER)

# Export processed data to csv & parquet
data_2024.write_csv(processed_csv_path)
data_2024.write_parquet(processed_parquet_path, compression="zstd", compression_level=16)

print(f"Took {time.time()-START_2024:.2f}s to finish processing the 2024 dataset.")
