import numpy as np
import pandas as pd
import os
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_curve)
import  matplotlib.pyplot as plt

from src.data.data_loader import load_data,split_data
from src.pipeline.pipeline import ModelPipeline
from src.utils.config import (
    DATA_PATH,TARGET_COLUMN,NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,TEST_SIZE,RANDOM_SIZE
)




def plot_roc_curve(y_test,y_scores):
    fpr,tpr,_ = roc_curve(y_test,y_scores)
    plt.figure()
    plt.plot(fpr,tpr,color="blue",label="ROC curve(area = %0.2f)" % roc_auc_score(y_test,y_scores))
    plt.plot([0,1],[0,1],color="red",linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("results/roc_curves/")  # Save the plot to a file
    plt.close()

def main():
    data = load_data(DATA_PATH)
    X= data.drop(TARGET_COLUMN,axis=1)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SIZE)

    

    pipeline = ModelPipeline(NUMERICAL_FEATURES,CATEGORICAL_FEATURES)
    
    pipeline.fit(X_train,y_train)

    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.pipeline.predict_proba(X_test)[:,1]

    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_scores)

    # Print the scores
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    plot_roc_curve(y_test,y_scores)

     # Save results to a file
    results = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    
    results_df = pd.DataFrame(results, index=[0])

    file_exists = os.path.isfile("results/results.csv")
        
    results_df.to_csv("results/results.csv",mode='a',header=not file_exists,index=False)
    # results_df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()