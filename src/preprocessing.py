import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()

    df['Over18'] = 1
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(
        df,
        columns=[
            'BusinessTravel','Department','EducationField',
            'JobRole','MaritalStatus','Gender'
        ],
        drop_first=True
    )

    num_cols = [
        'HourlyRate','EmployeeNumber','MonthlyIncome',
        'TotalWorkingYears','YearsAtCompany',
        'YearsInCurrentRole','YearsSinceLastPromotion',
        'YearsWithCurrManager','DailyRate','MonthlyRate'
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler
