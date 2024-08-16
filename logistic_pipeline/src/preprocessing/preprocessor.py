from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline



def create_preprocessor(numeric_features,categorical_features):
    numeric_transformer =  Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
        ('onehot',OneHotEncoder(handle_unknown = 'ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num',numeric_transformer, numeric_features),
            ('cat',categorical_transformer, categorical_features)
    ])

    return preprocessor

    