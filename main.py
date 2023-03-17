
import pickle
import pandas as pd
import warnings
from loan_default_engine_definition import BaseLoanDefaultModel

warnings.filterwarnings("ignore")


def load_model() -> BaseLoanDefaultModel:
    with open("./artefacts/loan_default_engine.pkl", "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    loan_default_engine = load_model()
    df = pd.read_csv("./data/credit.csv", index_col=0)
    decisions = df.apply(lambda row: loan_default_engine.decide(row), axis=1, result_type='expand')
    decisions.to_csv("credit_decisons.csv")
    print("Success!!")

