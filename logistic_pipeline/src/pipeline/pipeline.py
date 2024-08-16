from sklearn.pipeline import Pipeline
from src.preprocessing.preprocessor import create_preprocessor
from src.model.logistic_regression import create_model


def create_pipeline(numeric_features,categorical_features):
    return Pipeline([
        ('preprocessor',create_preprocessor(numeric_features,categorical_features)),
        ('classifier',create_model())

    ])

class ModelPipeline:
    def __init__(self, numeric_features, categorical_features):
        self.pipeline = create_pipeline(numeric_features,categorical_features)

    def fit(self, X, y):
        return self.pipeline.fit(X,y)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def score(self,X,y):
        return self.pipeline.score(X,y)