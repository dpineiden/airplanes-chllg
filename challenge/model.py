import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, TypeVar, Any
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from challenge.functions import process_dataframe
from challenge.features import complete_features

import joblib  # For saving and loading models


Model = TypeVar('Model')
BASE = Path(__file__).parent.absolute()

FEATURES_COLS = [
    "Fecha-I"   ,
    "Vlo-I"     ,
    "Ori-I"     ,
    "Des-I"     ,
    "Emp-I"     ,
    "Fecha-O"   ,
    "Vlo-O"     ,
    "Ori-O"     ,
    "Des-O"     ,
    "Emp-O"     ,
    "DIA"       ,
    "MES"       ,
    "AÑO"       ,
    "DIANOM"    ,
    "TIPOVUELO" ,
    "OPERA"     ,
    "SIGLAORI"  ,
    "SIGLADES"  ,
]

# Select relevant columns
categorical_columns = [
    'Ori-I', 'Des-I', 'Emp-I', 'Emp-O', 'DIANOM', 
    'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'period_day', "DIA",
    "MES"
]


@dataclass
class DelayModel:
    """
    This class builds the model LinearRegression with Balance
    given a set of features provided (optionally) on the constructor

    """
    features:list[str] = field(
        default_factory=lambda:FEATURES_COLS)
    _model:Optional[Model] = None
    random_state: int = 111
    model_file: Path = Path("delay_model.pkl")  # Default file to
    # save/load the model

    def __post_init__(
        self
    ):
        if not self._model:
            self._model = LogisticRegression(
                class_weight={0: 0.18381548426626987, 1:
                              0.8161845157337302})

    @property
    def model(self):
        return self._model

    def set_model(self, model:Any):
        self._model = model


    def save_model(self):
        """Save the trained model to disk."""
        with open(self.model_file, 'wb') as f:
            joblib.dump(self._model, f)
        print(f"Model saved to {self.model_file}")

    def load_model(self):
        """Load the model from disk."""
        if self.model_file.exists():
            with open(self.model_file, 'rb') as f:
                self._model = joblib.load(f)
            print(f"Model loaded from {self.model_file}")
        else:
            raise FileNotFoundError(f"Model file {self.model_file} not found!")



    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]| pd.DataFrame:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        process_dataframe(data)

        date_features = data[[
            "period_day", 
            "high_season",
            "min_diff", 
            "delay"]]


        # # Identify rows with NaN values
        # nan_rows = data_features[data_features.isna().any(axis=1)]

        # # Display rows with NaN values
        # print(nan_rows)

        # Apply get_dummies to categorical columns
        data['MES'] = pd.Categorical(data['MES'], categories=range(1, 13))

        df_categorical = pd.concat(
            [pd.get_dummies(data[field], prefix=field) 
             for field in categorical_columns], axis=1)

        year_data = data[["AÑO"]]
        # Concatenate numerical and dummy categorical features
        data_features = pd.concat([df_categorical, date_features, year_data], axis=1)

        target = None

        if target_column:

            data_features = data_features.dropna()

            target = data_features[[target_column]]
            data_features = data_features.drop([target_column],axis=1
                                               )
            data_features = complete_features(data_features)
            return data_features, target
        
        if "delay" in data_features.columns:
            data_features = data_features.drop(["delay"],axis=1 )

        data_features = complete_features(data_features)

        return data_features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        self.model.fit(features, target)
        return 

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        prediction = self.model.predict(features)
        return prediction.tolist()
