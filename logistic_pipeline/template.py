import os
from pathlib import Path

lst = [
    "src/data/__init__.py",
    "src/data/data_loader.py",

    "src/preprocessing/__init__.py",
    "src/preprocessing/preprocessor.py",

    "src/model/__init__.py",
    "src/model/logistic_regression.py",

    "src/pipeline/__init__.py",
    "src/pipeline/pipeline.py",

    "src/utils/__init__.py",
    "src/utils/config.py",
    
    "main.py",
    "requirements.txt"
]

for filepath in lst:
    filepath =Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass