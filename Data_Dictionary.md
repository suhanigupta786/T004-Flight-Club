<div align="center">

# ✈️ Flight Club
## Data Dictionary

**Flight Delay Prediction & Rescheduling**

*Dataset documentation • Feature definitions • Leakage prevention*

</div>

---

## 📌 1. Overview

The project uses 2013 US flight records to predict whether a flight will arrive more than 15 minutes late. The model is intended to make predictions using information available **before departure**.

> **Core rule:** A column used to calculate the target must not also be supplied as a model input.

### 🎯 Target definition

`delayed = 1` if `arr_delay > 15`; otherwise, `delayed = 0` if `arr_delay <= 15`.

- `1` — arrival delay is greater than 15 minutes.
- `0` — arrival delay is 15 minutes or less.
- Missing `arr_delay` values do not provide a valid target and should not automatically be labelled `0`.

---

## 🗂️ 2. Original Dataset Columns

> **Note:** These are standard fields from the 2013 US flights dataset. Verify each field against the actual `flights_clean.csv` before treating this as the final inventory.

| Column | Description | Typical data type | Project usage |
|---|---|---|---|
| `year` | Year in which the flight took place | Integer | Metadata; constant at 2013 |
| `month` | Month of the flight, from 1 to 12 | Integer | Feature |
| `day` | Day of the month | Integer | Used to construct flight date |
| `dep_time` | Actual departure time | Numeric / nullable | **Leakage — exclude** |
| `sched_dep_time` | Scheduled departure time | Integer | Feature |
| `dep_delay` | Difference between actual and scheduled departure, in minutes | Numeric / nullable | **Leakage — exclude** |
| `arr_time` | Actual arrival time | Numeric / nullable | **Leakage — exclude** |
| `sched_arr_time` | Scheduled arrival time | Integer | Feature |
| `arr_delay` | Difference between actual and scheduled arrival, in minutes | Numeric / nullable | Target source; **exclude from inputs** |
| `carrier` | Airline carrier code | String / categorical | Feature |
| `flight` | Flight number | Integer | Optional; review before use |
| `tailnum` | Aircraft registration identifier | String | Exclude initially |
| `origin` | Departure airport code | String / categorical | Feature |
| `dest` | Destination airport code | String / categorical | Feature |
| `air_time` | Time spent in the air, in minutes | Numeric / nullable | **Leakage — exclude** |
| `distance` | Flight distance, typically in miles | Numeric | Feature |
| `hour` | Hour component of scheduled departure | Integer | Feature |
| `minute` | Minute component of scheduled departure | Integer | Feature |
| `time_hour` | Hourly timestamp associated with the flight record | Datetime-like | Verify availability and use carefully |

---

## 🎯 3. Target and Project-Control Columns

| Column | Description | Typical data type | Project usage |
|---|---|---|---|
| `delayed` | Binary target derived from `arr_delay` using the 15-minute threshold | Integer (`0` or `1`) | **Prediction target** |
| `date` | Flight date constructed from year, month, and day | Date | Chronological ordering; not a direct feature by default |
| `split` | Existing train, validation, or test assignment | Categorical | Evaluation metadata; **not a feature** |
| `id` | Row identifier, if present | Integer / string | Tracking only; **exclude from inputs** |

**Target handling:** If `arr_delay` is missing, `delayed` cannot be reliably assigned using the target rule. Investigate or remove rows with missing target values rather than filling them with `0`.

---

## 🧪 4. Engineered Features

The following feature families are planned for the leakage-audit and feature-engineering stage. **Include a feature in the final dictionary only if it exists in the engineered CSV.**

| Feature / feature family | Description | Typical data type | Project usage |
|---|---|---|---|
| `origin_dest` | Combined origin and destination airport codes representing a route | Categorical | Feature |
| Departure-time sine/cosine | Cyclic encoding of scheduled departure time | Float | Feature |
| Weekday sine/cosine | Cyclic encoding of day of week | Float | Feature |
| Month sine/cosine | Cyclic encoding of month | Float | Feature |
| Scheduled-arrival-time sine/cosine | Cyclic encoding of scheduled arrival time | Float | Feature |
| Origin historical delay rate | Proportion of previously observed flights from an origin that were delayed | Float | Feature, past data only |
| Origin historical flight count | Number of previous flights observed for an origin | Integer | Feature |
| Carrier historical delay rate | Proportion of previously observed flights for a carrier that were delayed | Float | Feature, past data only |
| Carrier historical flight count | Number of previous flights observed for a carrier | Integer | Feature |
| Destination historical delay rate | Proportion of previously observed flights to a destination that were delayed | Float | Feature, past data only |
| Destination historical flight count | Number of previous flights observed for a destination | Integer | Feature |
| Route historical delay rate | Proportion of previously observed flights on a route that were delayed | Float | Feature, past data only |
| Route historical flight count | Number of previous flights observed on a route | Integer | Feature |

### 🕒 Historical feature rule

Historical features should use only the **seven preceding calendar dates**, excluding the current date and the current flight's outcome.

- Sort records by date before calculating historical statistics.
- Never use the current flight's `delayed` value to calculate its own predictors.
- If there are no earlier observations, a rate may be represented as `0` only when the corresponding history count is also retained.
- Ensure the same time-window logic is used consistently for training, validation, and testing.

---

## 🛡️ 5. Leakage Audit

Leakage occurs when a model receives information that would not be available at the time a prediction is meant to be made, or information that directly reveals the target.

| Column / field | Decision | Reason |
|---|---|---|
| `sched_dep_time`, `sched_arr_time` | **Keep** | Scheduled information |
| `month`, `day`, `hour`, `minute` | **Keep / engineer** | Calendar and schedule patterns |
| `carrier`, `origin`, `dest`, `distance` | **Keep** | Operational characteristics |
| Historical delay rates and counts | **Keep with safeguards** | Must be calculated from past observations only |
| `arr_delay` | **Exclude from model inputs** | Used to define the target |
| `dep_delay`, `dep_time`, `arr_time` | **Exclude** | Actual operational outcomes |
| `air_time` | **Exclude** | Known only after the flight |
| `id`, `split` | **Exclude from model inputs** | Identifier / evaluation metadata |
| `year` | **Usually exclude** | Constant in the 2013 dataset |
| `tailnum`, `flight` | **Review / exclude initially** | Identifier-like or high-cardinality fields |

> **Prediction-time assumption:** This dictionary assumes the prediction is made before departure. If the intended prediction time changes, review feature availability again.

---

## 🧹 6. Data Quality and Preprocessing Notes

- Check for missing values in each column.
- Do not replace missing target labels with `0`.
- Check for invalid or infinite numeric values.
- Fit imputers, scalers, and other learned preprocessing steps on the **training set only**.
- Apply the fitted preprocessing to validation and test sets without refitting.
- Preserve the existing chronological split labels.
- Keep `delayed`, `date`, `split`, and identifier fields separate from the model feature matrix unless a documented reason justifies otherwise.
- Verify that the final model feature list contains no leakage fields.

---

## 📁 7. Project Files

| File | Purpose |
|---|---|
| `flights_clean.csv` | Cleaned flight dataset |
| `03_Leakage_Audit_Feature_Engineering.ipynb` | Notebook documenting leakage checks and feature engineering |
| `flights_feature_engineered.csv` | Dataset containing engineered features |
| `03_leakage_audit.csv` | Record of leakage decisions |
| `03_feature_dictionary.csv` | Tabular data dictionary |
| `03_model_features.csv` | List of selected model features |

*File names and contents should be checked against the final repository before submission.*

---

## ✅ 8. Final Validation Checklist

- [ ] All documented columns exist in the final dataset.
- [ ] The `delayed` target follows the `arr_delay > 15` rule.
- [ ] Rows with missing targets are handled explicitly.
- [ ] Actual departure and arrival outcomes are excluded from model inputs.
- [ ] Historical features use past observations only.
- [ ] Existing train, validation, and test labels are preserved.
- [ ] Preprocessing is fitted on training data only.
- [ ] The model feature list excludes targets, identifiers, and split metadata.

---

<div align="center">

**Flight Club · Flight Delay Prediction & Rescheduling**

*Data Dictionary • Leakage-Aware Feature Engineering*

</div>
