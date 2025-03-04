# Hospital_Readmission_Project


### Project Description

This Git hub repository is group project that was completed as the final project for our MSAAI 540 program. This project was completed by Philip Felizarta, Vanessa Dyan Laxamana, and Parker Christenson. 

Our project was completed inside of Amazons Sagemaker AI platform, which utilized Jupyter notebooks. The project was completed in Python and utilized a variety of libraries such as Pandas, Numpy, Matplotlib, Seaborn, and Scikit-learn.


### File Structure

The repository is structured as follows:

- **EDA** - This folder contains the exploratory data analysis that was completed on the dataset, and also contains our models that were created.
- **Data** - This folder contains the dataset that was used for the project.
- **Figures** - This folder contains the figures that were created during the exploratory data analysis, and also contains our `Shap` library figures.
- **Models** - This folder contains the models that were created during the project.

- **Other Various files** This is a file that contains things such as, the final paper, encoding files, and other various files that were used during the project.

### Jupyter Notebooks

#### Pipeline Notebooks
- **CICD_Pipline** - The notebook implements a DAG CI/CD pipeline to read new feature data, automate feature engineering, train and tune a model, and deploy it to an AWS endpoint for monitoring. Shows DAG success and fail states.
- **Init_Data** - Passes and preprocesses .csv data to Athena tables
- **Setup_Feature_Store** - Queries Athena tables and passes data to Feature Stores for version control.
- **Train_XGB** - Splits training, test, and production data from the latest feature store. Conducts Feature Engineering using SHAP, trains a new XGBoost model, and passes the model to a Model-Store
- **Deployment_Monitoring** - Generates a mock api endpoint from the latest approved model. Creates a mock bias report after passing production data to the endpoint.

#### Misc Notebooks
- **EDA and random forest** - Basic exploratory data analysis and a trains a random forest model on the data.
- **EDA/Hospital_Re_admit_EDA** - Trains xgboost model and conducts EDA.

#### Misc Files
Any other files you see are intermediates creates by the CI/CD pipeline, these files are also saved in s3 buckets. They are in the repo for debugging purposes.
