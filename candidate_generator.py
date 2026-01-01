import random
import pandas as pd

FIRST_NAMES = [
    "Алексей", "Дмитрий", "Иван", "Никита", "Артем",
    "Максим", "Егор", "Кирилл", "Михаил", "Сергей"
]

LAST_NAMES = [
    "Иванов", "Петров", "Сидоров", "Смирнов", "Кузнецов",
    "Попов", "Лебедев", "Новиков", "Морозов", "Волков"
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

    experience = max(
        0,
        random.gauss(mu=age - 18, sigma=2)
    )

    experience = round(min(experience, age - 18), 1)

    skills = random.sample(
        SKILLS_POOL,
        random.randint(3, 7)
    )

    specialization = random.choice(SPECIALIZATIONS)

    expected_salary = random.randint(
        60000,
        300000
    )

    return {
        "Full Name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "Age": age,
        "Experience(years)": experience,
        "Specialization": specialization,
        "Skills": ", ".join(skills),
        "Has_Python": int("Python" in skills),
        "Has_SQL": int("SQL" in skills),
        "Skills_Count": len(skills),
        "Expected Salary": expected_salary
    }


def generate_dataset(n=500):
    return pd.DataFrame(
        [generate_candidate() for _ in range(n)]
    )


if __name__ == "__main__":
    df = generate_dataset(1000)
    df.to_excel("generated_candidates.xlsx", index=False)
    print("✅ Сгенерировано кандидатов:", len(df))
