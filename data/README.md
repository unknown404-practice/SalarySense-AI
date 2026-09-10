# Dataset Documentation — SalarySense AI

## Dataset Information
- **Dataset File:** `200000_employee_dataset.csv`
- **Total Records:** 200,000 employee compensation entries
- **Target Feature:** `Annual_Salary_USD` (Continuous annual salary in United States Dollars)

## Feature Schema

| Column Name | Data Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `Employee_ID` | String | Unique employee identifier (excluded during modeling) | e.g. `EMP000001` |
| `Education` | String / Category | Highest degree level completed | `Bachelor's`, `Diploma`, `High School`, `MBA`, `Master's`, `PhD` |
| `Working_Experience_Years` | Integer | Total years of relevant professional experience | 0 to 40 years |
| `Number_of_Projects` | Integer | Total projects completed by the employee | 0 to 80 projects |
| `Annual_Salary_USD` | Continuous Float | Annual compensation in US Dollars (**Target Variable**) | $18,000 to $288,900 |

## Placement Instructions
For security, repository size, and license compliance, large raw datasets (>5 MB) are omitted from git tracking via `.gitignore`. 

To run the offline training notebook or reproduction scripts locally:
1. Place `200000_employee_dataset.csv` in the project root directory or the `data/` folder.
2. Run `python export_model.py` or execute `notebooks/salary_prediction_analysis.ipynb`.
3. Note: The deployed Streamlit web application (`app.py`) runs independently using the pre-fitted lightweight pipeline (`models/salary_prediction_pipeline.joblib`, 3.9 KB) and does not require the raw 6 MB CSV file in production.
