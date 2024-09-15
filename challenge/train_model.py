from functions import read_data
from model import DelayModel,BASE
import typer
from pathlib import Path


app = typer.Typer()

@app.command()
def main_train(
        path:Path=BASE.parent / "data/data.csv", 
        model_name:str="delay_model"):
    model_name= f"{model_name}.pkl"
    # Assuming you have your training data
    df = read_data(path)  # Your data
    target_column = 'delay'

    delay_model = DelayModel(model_file=model_name)
    features, target = delay_model.preprocess(df, target_column)

    delay_model.fit(features, target)  # This will trai
    delay_model.save_model()

if __name__=="__main__":
    app()
