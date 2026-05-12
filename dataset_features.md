| Variable Category | Variable Name | Variable Meaning | Variable Code Notes |
| :-: | :-: | :-: | :-: |
| Demographics | year | Year of Interview | 2005-2024 |
| | age_group_00 | Age of the respondent | 13 point age group |
| | age_00 | Age of the respondant | `age_group_00` modified to reflect its ordinal nature for modelling |
| | sex_00 | Sex of the respondant |  |
| | race_00 | Race of the respondant | 8 point race category |
| | weight_00 | Weight of the respondent in pounds | Floating point number |
| | height_00 | Height of the respondent in inches | Floating point number |
| | bmi_00 | Body-Mass Index (BMI) of the respondent | [(`weight_00` * 703) / `height_00` ^ 2] 
| | marital_status_00 | Marital Status of the respondant | 6 point marital status category |
| | education_level_00 | Highest Education Level of the respondant | 6 point education level category |
| | education_00 | Highest Education Level of the respondant | `education_level_00` modified to reflect its ordinal nature for modelling |
| | employment_status_00 | Current Employment Status of the respondant | 8 point employment status category |
| | income_level_00 | Current Income Level of the respondant | 8 point income level category. Present for years 2005-2020 |
| | income_00 | Current Income Level of the respondant | `income_level_00` modified to reflect its ordinal nature for modelling |
| | income_level_01 | Current Income Level of the respondant | 11 point income level category. Present for years 2021-2024 |
| | income_01 | Current Income Level of the respondant | `income_level_01` modified to reflect its ordinal nature for modelling |
| Behavioral | general_health_00 | Respondent defined General Health Status | 5 point general health status |
| | gen_health_00 | Respondent defined General Health Status | `gen_health_00` modified to reflect its ordinal nature for modelling |
| | physical_health_days_00 | Poor Physical Health Days | Including physical illness and injury, for how many days during the past 30 days was the respondent's physical health not good? |
| | mental_health_days_00 | Poor Mental Health Days | Including stress, depression, and problems with emotions, for how many days during the past 30 days was the respondent's mental health not good? |
| | poor_health_days_00 | Poor Health Days | For about how many days did poor physical or mental health keep the respondent from doing their usual activities, such as self-care, work, or recreation? |
| | exercise_00 | Respondent Exercises? | During the past month, other than regular job, did the respondent participate in any physical activities or exercises? |
| | smoked_100_cigarettes_00 | Respondent Smokes? | Has the respondent smoked at least 100 cigarettes in their entire life?<br>[Note:  5 packs = 100 cigarettes] |
| | drinks_alcohol_00 | Respondent Drinks? | Adults who reported having had at least one drink of alcohol in the past 30 days |
| | avg_drinks_p_day_00 | Average Drinks per Day for past 30 days | During the past 30 days, on the days when the respondent drank, about how many drinks did they drink on the average?<br>(A 40 ounce beer would count as 3 drinks, or a cocktail drink with 2 shots would count as 2 drinks) |
| Clinical | high_bp_00 | Has High Blood Pressure? | 3 point blood pressure status. Present for odd-numbered years |
| | high_cholesterol_00 | Has High Cholesterol? | Binary cholesterol status. Present for odd-numbered years |
| | had_stroke_00 | Had a stroke? | Binary stroke status |
| | had_heart_attack_00 | Had a heart attack/myocardial infarction? | Binary heart attack status |
| | had_coronary_heart_disease_00 | Had angina/coronary heart disease? | Binary heart disease status |
| Healthcare Access | cost_barrier_00 | Cost Barrier to Checkup Present? | Was there a time in the past 12 months when the respondent needed to see a doctor but could not because they couldn't afford it? |
| | last_checkup_00 | Time since Last Routine Checkup | About how long has it been since the respondent last visited a doctor for a routine checkup?<br>[A routine checkup is a general physical exam, not an exam for a specific injury, illness, or condition.] |
| | l_checkup_00 | Time since Last Routine Checkup | `last_checkup_00` modified to reflect its ordinal nature for modelling |
| | has_personal_doctor_00 | Has Personal Healthcare Provider? | Does the respondent have one person or a group of doctors that they think of as their personal health care provider? |
| **TARGET** | diabetes | Diabetes Status of the Respondent | 3 class multiclass target. Can be converted into a binary target by merging class "Pre-diabetic" with class "No" |
