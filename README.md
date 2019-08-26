# Disaster Response Pipeline Project
This project is an app that analysis the added message and identify which it disaster relief categories. 
Also, This application include a visualization of the data.

# Folders:
1. data Include:
    - `disaster_messages.csv`, `disaster_categories.csv` the source files for messages and categories 
    - `process_data.py` use messages and categories files as an input and cleate a SQLite database containing a merged and cleaned dataset.
    - `DisasterResponse.db` The output SQLite file from `process_data.py`
    
2. models:
    - `train_classifier.py` use the data in `DisasterResponse.db` to train the model and print an evaluation of the mpdel
    - `classifier.pkl` thethe saved model from `train_classifier.py`. (not included in github because it is too big file)

3. app:
    - `run.py` to run the app file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
