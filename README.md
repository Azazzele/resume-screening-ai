Smart Candidate Filter
A machine learning-based system that automatically evaluates job candidates based on their resumes. It predicts whether a candidate is suitable for a position based on experience, age, and required technical skills (e.g., Python and SQL).

Overview
This is the first version of the project, so bugs and inconsistencies may occur.

Originally created to help me find a job, this project analyzed over 100,000 job listings and identified the best ones.

The system serves as a practical tool for candidate suitability prediction, leveraging machine learning and neural networks to analyze large datasets of applicants and highlight the most promising candidates.

Key Features
Data preprocessing including encoding categorical variables and skill extraction.

Customizable deep learning model architecture with multiple layers.

Capable of handling large datasets (100,000+ records).

Easily adaptable to different hiring criteria and skill sets.

Clear model evaluation metrics such as precision, recall, and F1-score.

Designed with real-world hiring challenges in mind, including handling cases where no candidates are selected.

Recommendations
Use a 4-layer deep learning model for training.

When selecting required skills, exclude the original 'Skills' object column and transform it into new binary columns (e.g., Has_Python, Has_SQL).

Be aware of a possible bug where the model might fail to select any candidates.

It is recommended to increase the number of training epochs up to 10,000 for better results.

How It Works
Data Input
Upload a spreadsheet (since.xlsx) containing candidate details.

Preprocessing

Label encode categorical fields (e.g., marital status, specialization).

Extract binary skill indicators (e.g., Has_Python, Has_SQL).

Drop unnecessary columns (e.g., personal info).

Model Training

Build a neural network using TensorFlow/Keras.

Train the model to classify candidates as Suitable or Not Suitable.

Prediction and Output

Model predicts candidate suitability on the test set.

Export the filtered list of top candidates for further review.

Technologies Used
Python 3.10

pandas==2.2.2

numpy==1.26.4

scikit-learn==1.5.0

tensorflow==2.15.0

torch==2.2.2

matplotlib==3.8.4

seaborn==0.13.2

openpyxl==3.1.2

Usage
bash
Копировать
Редактировать
# Install dependencies (preferably in a virtual environment)
pip install -r requirements.txt

# Run the main script
python main.py
