# Smart Candidate Filter

A machine learning-based system that automatically evaluates job candidates based on their resumes. It predicts whether a candidate is suitable for a position based on experience, age, and required technical skills (e.g., Python and SQL).

To begin with, I want to mention that this is the first version of the project, so bugs and inconsistencies are possible.

This project was originally created to help me find a job.
Thanks to the analysis, I reviewed over 100,000 job listings and identified the best ones.
This project is designed as a practical tool for candidate suitability prediction based on experience, skills, and other factors. It leverages machine learning and neural networks to analyze large datasets of job applicants and highlight the most promising candidates.

Key features:


Data preprocessing including encoding categorical variables and skill extraction.

Customizable deep learning model architecture with multiple layers.

Capability to handle large datasets (100,000+ records).

Easily adaptable to different hiring criteria and skill sets.

Clear metrics for model evaluation like precision, recall, and F1-score.

Designed with real-world hiring challenges in mind, including handling cases where no candidates are selected.

This project reflects a hands-on approach to solving real hiring challenges and can be used as a foundation for more advanced recruitment analytics tools.



I recommend using a 4-layer training model.
<img width="1712" height="496" alt="{5E85094E-9DE3-4F6F-85B0-B20E49AAAF52}" src="https://github.com/user-attachments/assets/06df65bb-7dd3-4490-ba1f-7ff08e929efb" />

When selecting the required skills from candidates, it is recommended to exclude this object and transform it accordingly.
<img width="839" height="49" alt="{FD71691E-3505-42BD-98C7-2E386DB62CB2}" src="https://github.com/user-attachments/assets/7cf1bb12-5b0e-4d14-9c9b-e84f80a58cac" />

a new column.
<img width="1093" height="112" alt="{B0C12620-EF57-42B6-A16C-675930ED83C2}" src="https://github.com/user-attachments/assets/3af1e26c-830b-4e95-9840-dbdc08c59c5f" />

There may be a bug where the model fails to select any candidates at all.
It is recommended to increase the number of epochs up to 10,000.


## Features

- ✅ Parses candidate data from Excel files
- ✅ Filters based on:
  - Experience (years)
  - Age
  - Required skills (Python, SQL, etc.)
- ✅ Builds a neural network classifier (Keras/TensorFlow)
- ✅ Outputs prediction results and filters top candidates
- ✅ Can be extended to use scikit-learn or PyTorch

## How It Works

1. **Data Input**  
   Upload a spreadsheet with candidate details (since.xlsx).

2. **Preprocessing**  
   - Label encoding for categorical fields (e.g. marital status)
   - Extract skills like Has_Python, Has_SQL
   - Drop unnecessary columns

3. **Model Training**  
   - Neural network built with TensorFlow
   - Trained to classify candidates as Suitable or Not Suitable

4. **Prediction Output**  
   - Model predicts suitability on test data
   - Suitable candidates exported for review

## Technologies Used

- Python 3.10  
- pandas==2.2.2  
- numpy==1.26.4  
- scikit-learn==1.5.0  
- tensorflow==2.15.0  
- torch==2.2.2  
- matplotlib==3.8.4  
- seaborn==0.13.2  
- openpyxl==3.1.2  
## Usage

bash
# Install requirements (you can also use a virtual environment)
pip install -r requirements.txt

# Run the model script
python main.py
Исправь возможноо что то ну или дополни
