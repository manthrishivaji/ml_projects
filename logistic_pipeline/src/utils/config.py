DATA_PATH = "Churn_modified.csv"
TARGET_COLUMN = "Exited"
NUMERICAL_FEATURES = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
CATEGORICAL_FEATURES = ['Geography','Gender']
TEST_SIZE = 0.2
RANDOM_SIZE = 42