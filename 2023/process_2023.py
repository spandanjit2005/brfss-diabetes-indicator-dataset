import polars as pl
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_csv_path = os.path.join(script_dir, '2023_BRFSS_RAW.csv')
cleaned_csv_path = os.path.join(script_dir, '2023_BRFSS_CLEANED.csv')

v01_2023_df = pl.read_csv(raw_csv_path)

# Extract & rename relevant columns for dataset

v02_2023_df = v01_2023_df.select([
    "SEXVAR", "_AGE_G", "WEIGHT2", "HEIGHT3", "EDUCA", "EMPLOY1", "INCOME3", "MARITAL",
    "PRIMINS1", "PERSDOC3", "MEDCOST1", "CHECKUP1", "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "_SMOKER3", "AVEDRNK3", "EXERANY2", "BPHIGH6", "BPMEDS1", "TOLDHI3", "CHOLMED3", 
    "DIABETE4"
    ])

new_columns = {
    "SEXVAR" : "SEX", "_AGE_G" : "AGE", "WEIGHT2" : "WGHT (lbs)", "HEIGHT3" : "HGHT (ft)", "EDUCA" : "EDUCATION_LEVEL", 
    "EMPLOY1" : "EMPLOYMENT_STATUS", "INCOME3" : "INCOME_LEVEL", "MARITAL" : "MARITAL_STATUS",
    "PRIMINS1" : "INSR_STATUS", "PERSDOC3" : "DCTR_STATUS", "MEDCOST1" : "COST_STATUS", "CHECKUP1" : "CHKP_STATUS", "GENHLTH" : "GEN_HLTH", "PHYSHLTH" : "PHYS_HLTH_DAYS", "MENTHLTH" : "MENT_HLTH_DAYS", "POORHLTH" : "POOR_HLTH_DAYS",
    "_SMOKER3" : "SMOK_STATUS", "AVEDRNK3" : "ALHL_STATUS", "EXERANY2" : "EXER", "BPHIGH6" : "HIGH_BP", "BPMEDS1" : "BP_MEDS", "TOLDHI3" : "HIGH_CHOL", "CHOLMED3" : "CHOL_MEDS",
    "DIABETE4" : "DIABETES_STATUS",
    }

v03_2023_df = v02_2023_df.rename(new_columns)

# Add `YEAR` = 2023

v04_2023_df = v03_2023_df.with_columns(
    pl.lit(2023).cast(pl.Int64).alias("YEAR")
)

# Process `SEX`
# Map of 0-1 scale:
# 0: Female
# 1: Male

v05_2023_df = v04_2023_df.with_columns(
    pl.col("SEX").map_elements(lambda x: 1 if x == 1 else 0).alias("SEX")
)

# Process `AGE`
# Map of 0-5 scale:
# 0: 18-24
# 1: 25-34
# 2: 35-44
# 3: 45-54
# 4: 55-64
# 5: 65+

AGE_mapping = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5
} 

v06_2023_df = v05_2023_df.with_columns(
    pl.col("AGE").replace_strict(AGE_mapping, default=None).cast(pl.Int64).alias("AGE")
)

# Process `WEIGHT (lbs)`
# Convert from kg to lbs where necessary

def normalize_weight(x):

    if 50 <= x <= 766:
        return x

    if 9023 <= x <= 9352:
        kg = x - 9000
        return round(kg * 2.20462)
    if x == 7777 or x == 9999:
        return None
        
    return None

v07_2023_df = v06_2023_df.with_columns(
    pl.col("WGHT (lbs)").map_elements(normalize_weight).alias("WGHT (lbs)")
)

# Process `HEIGHT (ft)`
# Convert from cm to ft where necessary
# Remove outliers (> 9 ft)

def normalize_height(x):

    if 200 <= x <= 711:
        feet = x // 100
        inches = x % 100
        return round(feet + inches / 12, 2)

    if 9061 <= x <= 9998:
        centimeters = x - 9000
        return round(centimeters / 30.48, 2)

    if x == 7777 or x == 9999:
        return None
        
    return None

v08_2023_df = v07_2023_df.with_columns(
    pl.col("HGHT (ft)").map_elements(normalize_height).alias("HGHT (ft)")
)

v09_2023_df = v08_2023_df.filter(
    (pl.col("HGHT (ft)") <= 9)
)

# Calculate `BMI`

def calculate_bmi(row):
    weight = row["WGHT (lbs)"]
    height = row["HGHT (ft)"]

    if weight is None or height is None or height == 0:
        return None

    height *= 12

    bmi = (weight / (height * height)) * 703
    return round(bmi, 2)

v10_2023_df = v09_2023_df.with_columns(
    pl.struct(["WGHT (lbs)", "HGHT (ft)"]).map_elements(calculate_bmi).alias("BMI")
)

# Process `EDUCATION_LEVEL`
# Map of 0-5 scale:
# 0: Never attended school or only kindergarten
# 1: Grades 1 through 8 (Elementary)
# 2: Grades 9 through 11 (Some high school)
# 3: Grade 12 or GED (High school graduate)
# 4: College 1 year to 3 years (Some college or technical school)
# 5: College 4 years or more (College graduate)

EDUCATION_LEVEL_mapping = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
}

v11_2023_df = v10_2023_df.with_columns(
    pl.col("EDUCATION_LEVEL").replace_strict(EDUCATION_LEVEL_mapping, default=None).alias("EDUCATION_LEVEL")
)

# Process `EMPLOYMENT_STATUS`
# Map of 0-7 scale:
# 0: Unable to work
# 1: Employed for wages
# 2: Self-employed
# 3: Out of work for 1 year or more
# 4: Out of work for less than 1 year
# 5: A homemaker
# 6: A student
# 7: Retired

EMPLOYMENT_STATUS_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 0, 9: None
}

v12_2023_df = v11_2023_df.with_columns(
    pl.col("EMPLOYMENT_STATUS").replace_strict(EMPLOYMENT_STATUS_mapping, default=None).alias("EMPLOYMENT_STATUS")
)

# Process `INCOME_LEVEL`
# Map of 1-11 scale:
# 1: Less than $10,000
# 2: $10,000 to less than $15,000
# 3: $15,000 to less than $20,000
# 4: $20,000 to less than $25,000
# 5: $25,000 to less than $35,000
# 6: $35,000 to less than $50,000
# 7: $50,000 to less than $75,000
# 8: $75,000 to less than $100,000
# 9: $100,000 to less than $150,000
# 10: $150,000 to less than $200,000
# 11: $200,000 or more

INCOME_LEVEL_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 77: None, 99: None
}

v13_2023_df = v12_2023_df.with_columns(
    pl.col("INCOME_LEVEL").cast(pl.Int64).replace_strict(INCOME_LEVEL_mapping, default=None).cast(pl.Int64, strict=False).alias("INCOME_LEVEL")
)

# Process `MARITAL_STATUS`
# Map of 1-6 scale:
# 1: Married
# 2: Divorced
# 3: Widowed
# 4: Separated
# 5: Never married
# 6: A member of an unmarried couple

MARITAL_STATUS_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: None
}

v14_2023_df = v13_2023_df.with_columns(
    pl.col("MARITAL_STATUS").replace_strict(MARITAL_STATUS_mapping, default=None).alias("MARITAL_STATUS")
)

# Process `INSR_STATUS`
# Map of 0-10 scale:
# 0: No coverage of any type
# 1: A plan purchased through an employer or union (including plans purchased through another person's employer)
# 2: A private nongovernmental plan that you or another family member buys on your own
# 3: Medicare
# 4: Medigap
# 5: Medicaid
# 6: Children's Health Insurance Program (CHIP)
# 7: Military related health care: TRICARE (CHAMPUS) / VA health care / CHAMP- VA
# 8: Indian Health Service
# 9: State sponsored health plan
# 10: Other government program

def normalize_insurance(x):

    if 1 <= x <= 10:
        return int(x)

    if x == 88:
        return 0

    if x == 77 or x == 99:
        return None
        
    return None

v15_2023_df = v14_2023_df.with_columns(
    pl.col("INSR_STATUS").map_elements(normalize_insurance).alias("INSR_STATUS")
)

# Process `DCTR_STATUS`
# Map of 0-2 scale:
# 0: No
# 1: Yes, only one
# 2: Yes, more than one

DCTR_STATUS_mapping = {
    1: 1, 2: 2, 3: 0, 7: None, 9: None
}

v16_2023_df = v15_2023_df.with_columns(
    pl.col("DCTR_STATUS").replace_strict(DCTR_STATUS_mapping, default=None).alias("DCTR_STATUS")
)

# Process `COST_STATUS`
# Map of 0-1 scale:
# 0: No
# 1: Yes

COST_STATUS_mapping = {
    1: 1, 2: 0, 7: None, 9: None
}

v17_2023_df = v16_2023_df.with_columns(
    pl.col("COST_STATUS").replace_strict(COST_STATUS_mapping, default=None).alias("COST_STATUS")
)

# Process `CHKP_STATUS`
# Map of 0-4 scale:
# 0: Never
# 1: Within past year (anytime less than 12 months ago)
# 2: Within past 2 years (1 year but less than 2 years ago)
# 3: Within past 5 years (2 years but less than 5 years ago
# 4: 5 or more years ago

CHKP_STATUS_mapping = {
    1: 1, 2: 2, 3: 3, 4: 4, 8: 0, 7: None, 9: None
}

v18_2023_df = v17_2023_df.with_columns(
    pl.col("CHKP_STATUS").replace_strict(CHKP_STATUS_mapping, default=None).alias("CHKP_STATUS")
)

# Process `GEN_HLTH`
# Map of 1-5 scale:
# 5: Excellent
# 4: Very good
# 3: Good
# 2: Fair
# 1: Poor

GEN_HLTH_mapping = {
    1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 7: None, 9: None
}

v19_2023_df = v18_2023_df.with_columns(
    pl.col("GEN_HLTH").replace_strict(GEN_HLTH_mapping, default=None).alias("GEN_HLTH")
)

# Process `PHYS_HLTH_DAYS`, `MENT_HLTH_DAYS`, `POOR_HLTH_DAYS`
# Map of 0-30 scale:
# 0-30: Number of days of poor physical/mental/general health in past 30 days

def normmalize_health_days(x):

    if 0 <= x <= 30:
        return int(x)

    if x == 88:
        return 0

    if x == 77 or x == 99:
        return None
        
    return None

v20_2023_df = v19_2023_df.with_columns(
    pl.col("PHYS_HLTH_DAYS").map_elements(normmalize_health_days).alias("PHYS_HLTH_DAYS"),
    pl.col("MENT_HLTH_DAYS").map_elements(normmalize_health_days).alias("MENT_HLTH_DAYS"),
    pl.col("POOR_HLTH_DAYS").map_elements(normmalize_health_days).alias("POOR_HLTH_DAYS")
)

# Process `SMOK_STATUS`
# Map of 0-2 scale:
# 0: Never smoked
# 1: Former smoker
# 2: Current smoker (some days)
# 3: Current smoker (everyday)

SMOKE_STATUS_mapping = {
    1: 3, 2: 2, 3: 1, 4: 0, 7: None, 9: None
}

v21_2023_df = v20_2023_df.with_columns(
    pl.col("SMOK_STATUS").replace_strict(SMOKE_STATUS_mapping, default=None).alias("SMOK_STATUS")
)

# Process `ALHL_STATUS`
# Map of 0-76 scale:
# 0-76: Number of drinks per week

def normalize_alcohol(x):

    if 1 <= x <= 76:
        return int(x)

    if x == 88:
        return 0

    if x == 77 or x == 99:
        return None
        
    return None

v22_2023_df = v21_2023_df.with_columns(
    pl.col("ALHL_STATUS").map_elements(normalize_alcohol).alias("ALHL_STATUS")
)

# Process `EXER`
# Map of 0-1 scale:
# 0: No
# 1: Yes

EXCR_mapping = {
    1: 1, 2: 0, 7: None, 9: None
}

v23_2023_df = v22_2023_df.with_columns(
    pl.col("EXER").replace_strict(EXCR_mapping, default=None).alias("EXER")
)

# Process `HIGH_BP`
# Map of 0-3 scale:
# 0: No
# 1: Told borderline high or pre-hypertensive or elevated blood pressure
# 2: Yes, but female told only during pregnancy
# 3: Yes

HIGH_BP_mapping = {
    1: 3, 2: 2, 3: 0, 4: 1, 7: None, 9: None
}

v24_2023_df = v23_2023_df.with_columns(
    pl.col("HIGH_BP").replace_strict(HIGH_BP_mapping, default=None).alias("HIGH_BP")
)

# Process `BP_MEDS`
# Map of 0-1 scale:
# 0: No
# 1: Yes

BP_MEDS_mapping = {
    1: 1, 2: 0, 7: None, 9: None
}

v25_2023_df = v24_2023_df.with_columns(
    pl.col("BP_MEDS").replace_strict(BP_MEDS_mapping, default=None).alias("BP_MEDS")
)

# Process `HIGH_CHOL`
# Map of 0-1 scale:
# 0: No
# 1: Yes

HIGH_CHOL_mapping = {
    1: 1, 2: 0, 7: None, 9: None
}

v26_2023_df = v25_2023_df.with_columns(
    pl.col("HIGH_CHOL").replace_strict(HIGH_CHOL_mapping, default=None).alias("HIGH_CHOL")
)

# Process `CHOL_MEDS`
# Map of 0-1 scale:
# 0: No
# 1: Yes

CHOL_MEDS_mapping = {
    1: 1, 2: 0, 7: None, 9: None
}

v27_2023_df = v26_2023_df.with_columns(
    pl.col("CHOL_MEDS").replace_strict(CHOL_MEDS_mapping, default=None).alias("CHOL_MEDS")
)

# Process `DIABETES_STATUS`
# Map of 0-3 scale:
# 0: No
# 1: No, pre-diabetes or borderline diabetes
# 2: Yes, but female told only during pregnancy
# 3: Yes

DIABETES_STATUS_mapping = {
    1: 3, 2: 2, 3: 0, 4: 1, 7: None, 9: None
}

v28_2023_df = v27_2023_df.with_columns(
    pl.col("DIABETES_STATUS").replace_strict(DIABETES_STATUS_mapping, default=None).alias("DIABETES_STATUS")
)

# Arrange columns properly

v29_2023_df = v28_2023_df.select([
    "YEAR", "SEX", "AGE", "WGHT (lbs)", "HGHT (ft)", "BMI", "EDUCATION_LEVEL", "EMPLOYMENT_STATUS", "INCOME_LEVEL", "MARITAL_STATUS",
    "INSR_STATUS", "DCTR_STATUS", "COST_STATUS", "CHKP_STATUS", "GEN_HLTH", "PHYS_HLTH_DAYS", "MENT_HLTH_DAYS", "POOR_HLTH_DAYS",
    "SMOK_STATUS", "ALHL_STATUS", "EXER", "HIGH_BP", "BP_MEDS", "HIGH_CHOL", "CHOL_MEDS",
    "DIABETES_STATUS"
])

# Drop rows with null values in critical columns

v30_2023_df = v29_2023_df.drop_nulls(subset=[
    "DIABETES_STATUS", "SEX", "AGE", "WGHT (lbs)", "HGHT (ft)", "BMI"
])

# Export cleaned dataframe to CSV

v30_2023_df.write_csv(cleaned_csv_path)