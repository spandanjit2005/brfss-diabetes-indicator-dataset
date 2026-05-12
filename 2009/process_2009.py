import os
import time
import polars as pl

START_2009 = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_csv_path = os.path.join(script_dir, "2009_BRFSS_RAW.csv")

processed_csv_path = os.path.join(script_dir, "DATASET_2009.csv")
processed_parquet_path = os.path.join(script_dir, "DATASET_2009.parquet")

data_2009 = pl.read_csv(raw_csv_path, ignore_errors=True)

# Extract and rename relevant features
data_2009 = data_2009.select([
    # Demographics
    "_AGEG5YR", "SEX", "RACE2",
    "MARITAL", "EDUCA", "EMPLOY", "INCOME2", "WEIGHT2", "HEIGHT3",

    # Behavioral
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "EXERANY2", "SMOKE100", "DRNKANY4", "AVEDRNK2",

    # Clinical
    "BPHIGH4", "TOLDHI2", "CVDSTRK3", "CVDINFR4", "CVDCRHD4",

    # Healthcare Access
    "MEDCOST", "CHECKUP1", "PERSDOC2",

    # Target
    "DIABETE2"
])

NEW_FEATURE_NAME_map = {
    # Demographics
    "_AGEG5YR": "age_group_00", "SEX": "sex_00", "RACE2": "race_00",
    "WEIGHT2": "weight_00", "HEIGHT3": "height_00", "MARITAL": "marital_status_00",
    "EDUCA": "education_level_00", "EMPLOY": "employment_status_00", "INCOME2": "income_level_00",

    # Behavioral
    "GENHLTH": "general_health_00", "PHYSHLTH": "physical_health_days_00", "MENTHLTH": "mental_health_days_00",
    "POORHLTH": "poor_health_days_00", "EXERANY2": "exercise_00", "SMOKE100": "smoked_100_cigarettes_00",
    "DRNKANY4": "drinks_alcohol_00", "AVEDRNK2": "avg_drinks_p_day_00",

    # Clinical
    "BPHIGH4": "high_bp_00", "TOLDHI2": "high_cholesterol_00",
    "CVDSTRK3": "had_stroke_00", "CVDINFR4": "had_heart_attack_00", "CVDCRHD4": "had_coronary_heart_disease_00",

    # Healthcare Access
    "MEDCOST": "cost_barrier_00", "CHECKUP1": "last_checkup_00", "PERSDOC2": "has_personal_doctor_00",

    # Target
    "DIABETE2": "diabetes"
}

data_2009 = data_2009.rename(NEW_FEATURE_NAME_map)

# Cast columns into specific data types & define `BINARY_map`
INT_8_cols = [
    "age_group_00", "sex_00", "race_00", 
    "marital_status_00", "education_level_00", "employment_status_00", "income_level_00",

    "general_health_00", "physical_health_days_00", "mental_health_days_00", 
    "exercise_00", "smoked_100_cigarettes_00", 

    "high_bp_00", "had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00",

    "cost_barrier_00", "last_checkup_00", "has_personal_doctor_00",

    "diabetes"
]

data_2009 = data_2009.with_columns([
    pl.col(col).cast(pl.Int8, strict=False)
    for col in INT_8_cols
])

FLOAT_64_cols = [
    "weight_00", "height_00", 
    "poor_health_days_00", "drinks_alcohol_00", "avg_drinks_p_day_00",
    "high_cholesterol_00"
]

data_2009 = data_2009.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in FLOAT_64_cols
])

BINARY_map = {
    1: True,
    2: False
}

# Add `year` = 2009
data_2009 = data_2009.with_columns(pl.lit(2009).cast(pl.Int64).alias("year"))

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

data_2009 = data_2009.with_columns(
    pl.when((pl.col("age_group_00") >= 1) & (pl.col("age_group_00") <= 13)).then(pl.col("age_group_00"))
    .otherwise(None)
    .alias("age_00")
)

data_2009 = data_2009.with_columns(
    pl.col("age_group_00").replace_strict(AGE_GROUP_map, default=None)
    .cast(pl.Categorical)
    .alias("age_group_00")
)

# Process `sex_00`
data_2009 = data_2009.with_columns(
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

data_2009 = data_2009.with_columns(
    pl.col("race_00").replace_strict(RACE_map, default=None)
    .cast(pl.Categorical)
    .alias("race_00")
)

# Process `weight_00`
data_2009 = data_2009.with_columns(
    pl.when((pl.col("weight_00") >= 50) & (pl.col("weight_00") <= 999))
    .then(pl.col("weight_00"))

    .when((pl.col("weight_00") >= 9000) & (pl.col("weight_00") <= 9998))
    .then(((pl.col("weight_00") - 9000) * 2.20462).round(2))

    .otherwise(None)
    .cast(pl.Float64)
    .alias("weight_00")
)

# Process `height_00`
data_2009 = data_2009.with_columns(
    pl.when((pl.col("height_00") >= 200) & (pl.col("height_00") <= 711))
    .then((pl.col("height_00") // 100) * 12 + (pl.col("height_00") % 100))

    .when((pl.col("height_00") >= 9000) & (pl.col("height_00") <= 9998))
    .then(((pl.col("height_00") - 9000) / 2.54).round(2))

    .otherwise(None)
    .cast(pl.Float64)
    .alias("height_00")
)

# Compute `bmi`
data_2009 = data_2009.with_columns(
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

data_2009 = data_2009.with_columns(
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

data_2009 = data_2009.with_columns(
    pl.when((pl.col("education_level_00") >= 1) & (pl.col("education_level_00") <= 6))
    .then(pl.col("education_level_00"))

    .otherwise(None)
    .alias("education_00")
)

data_2009 = data_2009.with_columns(
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

data_2009 = data_2009.with_columns(
    pl.col("employment_status_00").replace_strict(EMPLOYMENT_STATUS_map, default=None)
    .cast(pl.Categorical)
    .alias("employment_status_00")
)

# Process `income_level_00`, & create `income_00` for modelling
INCOME_LEVEL_map = {
    1: "Less than $10,000",
    2: "$10,000 to less than $15,000",
    3: "$15,000 to less than $20,000",
    4: "$20,000 to less than $25,000",
    5: "$25,000 to less than $35,000",
    6: "$35,000 to less than $50,000",
    7: "$50,000 to less than $75,000",
    8: "$75,000 or more"
}

data_2009 = data_2009.with_columns(
    pl.when((pl.col("income_level_00") >= 1) & (pl.col("income_level_00") <= 8))
    .then(pl.col("income_level_00"))

    .otherwise(None)
    .alias("income_00")
)

data_2009 = data_2009.with_columns(
    pl.col("income_level_00").replace_strict(INCOME_LEVEL_map, default=None)
    .cast(pl.Categorical)
    .alias("income_level_00")
)

# Process `general_health_00`, & create `gen_health_00` for modelling
GENERAL_HEALTH_map = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

data_2009 = data_2009.with_columns(
    pl.when((pl.col("general_health_00") >= 1) & (pl.col("general_health_00") <= 5))
    .then(pl.col("general_health_00"))

    .otherwise(None)
    .alias("gen_health_00")
)

data_2009 = data_2009.with_columns(
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

data_2009 = data_2009.with_columns(
    [process_health_days(col) for col in HEALTH_DAY_cols]
)

# Process `exercise_00`, `smoked_100_cigarettes_00`, & `drinks_alcohol_00`
BINARY_BEHAVIORAL_cols = ["exercise_00", "smoked_100_cigarettes_00", "drinks_alcohol_00"]

data_2009 = data_2009.with_columns([
    pl.col(col).replace_strict(BINARY_map, default=None)
    .cast(pl.Boolean)
    .alias(col)
    for col in BINARY_BEHAVIORAL_cols
])

# Process `avg_drinks_p_day_00`
data_2009 = data_2009.with_columns(
    pl.when(((pl.col("avg_drinks_p_day_00") >= 1) & (pl.col("avg_drinks_p_day_00") <= 76)))
    .then(pl.col("avg_drinks_p_day_00"))

    .otherwise(None)
    .cast(pl.Int8)
    .alias("avg_drinks_p_day_00")
)

# Process `high_bp_00`
HIGH_BP_map = {
    1: "Yes",
    2: "Yes",
    3: "No",
    4: "Borderline"
}

data_2009 = data_2009.with_columns(
    pl.col("high_bp_00").replace_strict(HIGH_BP_map, default=None)
    .cast(pl.Categorical)
    .alias("high_bp_00")
)

# Process `high_cholesterol_00`, `had_stroke_00`, `had_heart_attack_00`, & had_coronary_heart_disease_00
BINARY_CLINICAL_cols = ["high_cholesterol_00", "had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00"]

data_2009 = data_2009.with_columns([
    pl.col(col).replace_strict(BINARY_map, default=None)
    .cast(pl.Boolean)
    .alias(col)
    for col in BINARY_CLINICAL_cols
])

# Process `cost_barrier_00`
data_2009 = data_2009.with_columns([
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

data_2009 = data_2009.with_columns(
    pl.when((pl.col("last_checkup_00") >= 1) & (pl.col("last_checkup_00") <= 4))
    .then(pl.col("last_checkup_00"))

    .when(pl.col("last_checkup_00") == 8).then(pl.lit(5))

    .otherwise(None)
    .alias("l_checkup_00")
)

data_2009 = data_2009.with_columns([
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

data_2009 = data_2009.with_columns([
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

data_2009 = data_2009.with_columns([
    pl.col("diabetes").replace_strict(DIABETES_map, default=None)
    .cast(pl.Categorical)
    .alias("diabetes")
])

# Drop rows with no usable data for the feature column
data_2009 = data_2009.filter(pl.col("diabetes").is_not_null())

# Drop duplicates across all columns
data_2009 = data_2009.unique(keep="first")

# Order the features appropriately
FEATURE_ORDER = [
    # Demographics
    "year", "age_group_00", "age_00", "sex_00", "race_00",
    "weight_00", "height_00", "bmi_00", "marital_status_00",
    "education_level_00", "education_00", "employment_status_00", "income_level_00", "income_00",

    # Behavioral
    "general_health_00", "gen_health_00", "physical_health_days_00", "mental_health_days_00",
    "poor_health_days_00", "exercise_00", "smoked_100_cigarettes_00",
    "drinks_alcohol_00", "avg_drinks_p_day_00",

    # Clinical
    "high_bp_00", "high_cholesterol_00",
    "had_stroke_00", "had_heart_attack_00", "had_coronary_heart_disease_00",

    # Healthcare Access
    "cost_barrier_00", "last_checkup_00", "l_checkup_00", "has_personal_doctor_00",

    # Target
    "diabetes"
]

data_2009 = data_2009.select(FEATURE_ORDER)

# Export processed data to csv & parquet
data_2009.write_csv(processed_csv_path)
data_2009.write_parquet(processed_parquet_path, compression="zstd", compression_level=16)

print(f"Took {time.time()-START_2009:.2f}s to finish processing the 2009 dataset.")
