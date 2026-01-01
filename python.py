import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib

# =========================
# 1. ГЕНЕРАТОР КАНДИДАТОВ
# =========================

FIRST_NAMES = [
    "Alex", "Dmitry", "Ivan", "Nikita", "Artem",
    "Maxim", "Egor", "Kirill", "Mikhail", "Sergey"
]

LAST_NAMES = [
    "Ivanov", "Petrov", "Sidorov", "Smirnov", "Kuznetsov",
    "Popov", "Lebedev", "Novikov", "Morozov", "Volkov"
]

SPECIALIZATIONS = [
    "Backend Developer",
    "Frontend Developer",
    "Fullstack Developer",
    "Data Scientist",
    "ML Engineer",
    "DevOps",
    "QA Engineer"
]

SKILLS_POOL = [
    "Python", "SQL", "Django", "Flask", "FastAPI",
    "PostgreSQL", "Docker", "Git", "Linux",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow"
]


def generate_candidate():
    age = random.randint(20, 50)
    experience = round(min(max(random.gauss(age - 18, 2), 0), age - 18), 1)

    skills = random.sample(SKILLS_POOL, random.randint(3, 7))

    expected_salary = random.randint(60000, 300000)

    return {
        "Full Name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "Age": age,
        "Experience(years)": experience,
        "Specialization": random.choice(SPECIALIZATIONS),
        "Skills": ", ".join(skills),
        "Has_Python": int("Python" in skills),
        "Has_SQL": int("SQL" in skills),
        "Skills_Count": len(skills),
        "Expected Salary": expected_salary
    }


def generate_dataset(n=1000):
    return pd.DataFrame([generate_candidate() for _ in range(n)])


# =========================
# 2. HR SCORE (логика HR)
# =========================

def calculate_hr_score(row):
    score = 0

    score += min(row["Experience(years)"] * 8, 40)
    score += row["Has_Python"] * 25
    score += row["Has_SQL"] * 10
    score += min(row["Skills_Count"] * 3, 15)

    if row["Expected Salary"] > 200000:
        score -= 15
    elif row["Expected Salary"] > 150000:
        score -= 8

    if row["Age"] < 23 or row["Age"] > 45:
        score -= 5

    return max(0, min(100, int(score)))


# =========================
# 3. MAIN PIPELINE
# =========================

data = generate_dataset(1000)
print(f"Generated candidates: {len(data)}")

data["HR_Score"] = data.apply(calculate_hr_score, axis=1)
data["Suitable"] = (data["HR_Score"] >= 60).astype(int)

# шум (чтобы было реалистично)
noise = np.random.rand(len(data)) < 0.1
data.loc[noise, "Suitable"] ^= 1

# encoding
le = LabelEncoder()
data["Specialization"] = le.fit_transform(data["Specialization"])

# features
X = data[
    [
        "Age",
        "Experience(years)",
        "Expected Salary",
        "Has_Python",
        "Has_SQL",
        "Skills_Count",
        "Specialization"
    ]
]
y = data["Suitable"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

test_idx = X_test.index

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. MODEL
# =========================

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# importance
print("\nFEATURE IMPORTANCE:")
for name, imp in sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:20s} -> {imp:.3f}")

# evaluation
y_pred = model.predict(X_test)
print("\nGRADIENT BOOSTING RESULT")
print(classification_report(y_test, y_pred))

# =========================
# 5. FINAL CANDIDATES
# =========================

test_data = data.loc[test_idx].copy()
test_data["Hire_Probability"] = model.predict_proba(X_test)[:, 1]
test_data["Predicted_Suitable"] = y_pred

final_candidates = (
    test_data[test_data["Predicted_Suitable"] == 1]
    .sort_values("Hire_Probability", ascending=False)
    [
        [
            "Full Name",
            "Age",
            "Experience(years)",
            "Expected Salary",
            "HR_Score",
            "Hire_Probability",
            "Skills"
        ]
    ]
)

print("\nFINAL CANDIDATES:")
print(final_candidates.head(10))

final_candidates.to_excel("selected_candidates.xlsx", index=False)

# =========================
# 6. RANDOM FOREST (compare)
# =========================

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST RESULT")
print(classification_report(y_test, rf_pred))

# =========================
# 7. SAVE MODEL
# =========================

joblib.dump(model, "candidate_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "specialization_encoder.pkl")

print("DONE")
print("Files created:")
print("- selected_candidates.xlsx")
print("- candidate_model.pkl")
print("- scaler.pkl")
print("- specialization_encoder.pkl")
