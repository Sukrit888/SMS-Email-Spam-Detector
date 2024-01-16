from pickle import load

# Helper Functio nto Load Models
def load_model(file_path: str):
    return load(open(file_path,'rb'))