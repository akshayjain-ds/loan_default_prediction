
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from category_encoders.glmm import GLMMEncoder
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import shap
from collections import OrderedDict
from enum import Enum

FeatureNames = Tuple[str, ...]
EngineInputOutputT = Dict[str, Any]
DataFrameInputT = pd.DataFrame


class RiskCategory(Enum):
    approved = 'APPROVED'
    mkyc = 'MKYC'
    reject = 'REJECT'


def dict_to_dataframe(payload: EngineInputOutputT) -> DataFrameInputT:
    """Convert given payload to a pandas DataFrame and return that."""
    return pd.DataFrame(payload, index=[0])


def round_to_int(number):
    return int(round(number))


class LoadDefaultModelError(Exception):
    pass


def month_year_feature_transform_to_months(x: str) -> int:

    if pd.isnull(x):
        return np.nan
    else:
        time_x, timeunit = str(x).split()
        if timeunit == 'months':
            return int(time_x)
        elif timeunit == 'years':
            return int(time_x) * 12
        else:
            return np.nan


class BaseLoanDefaultModel:

    def __init__(self,
                 input_feature_names,
                 decision_feature_names,
                 **kwargs):

        self.kwargs = kwargs

        self.MODEL_VERSION = '1.0.0'
        self.input_feature_names = tuple(input_feature_names)
        self.decision_feature_names = tuple(decision_feature_names)

        self.target_encoder = GLMMEncoder(cols=[])

        self.model_class = xgb.XGBClassifier
        self.caliberation_class = CalibratedClassifierCV
        self.shap_explainer = shap.TreeExplainer
        self.model_obj = None
        self.model_calibration_obj = None
        self.shap_explainer_obj = None

        self.THRESHOLD_APPROVED_MKYC = float('nan')
        self.THRESHOLD_MKYC_REJECT = float('nan')

        # store risk category gradients
        self.RISK_GRADIENT_LOW = float('nan')
        self.RISK_GRADIENT_HIGH = float('nan')

        self.MIN_RISK_SCORE = 0
        self.MAX_RISK_SCORE = 1000
        self.TM_THRESHOLD_ABSOLUTE = 200

        self.RISK_SCORE_MULTIPLIER = 1000
        self.TM_THRESHOLD = None

        self.FAILSAFE_RISK_SCORE = 1000
        self.training_complete = False

    def transform_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        """
        decision_features = input_data[[*self.input_feature_names]].copy()

        decision_features['personal_status'] = decision_features[['personal_status']].apply(lambda x: x.fillna("unknown"), axis=1)
        decision_features['employment_length_months'] = decision_features['employment_length'].apply(
            lambda x: month_year_feature_transform_to_months(x))
        decision_features['residence_history_months'] = decision_features['residence_history'].apply(
            lambda x: month_year_feature_transform_to_months(x))

        decision_features[[*self.decision_feature_names]] = self.target_encoder.transform(
            decision_features[[*self.decision_feature_names]])

        return decision_features[[*self.decision_feature_names]]

    def shap_reason(self, feature_df: pd.DataFrame) -> pd.DataFrame:

        shap_values = self.shap_explainer_obj.shap_values(feature_df)

        if isinstance(shap_values, np.ndarray):
            shap_values = pd.DataFrame(shap_values, index=feature_df.index, columns=[*self.decision_feature_names])
        else:
            shap_values = pd.DataFrame(shap_values.values, index=feature_df.index, columns=[*self.decision_feature_names])

        reason = shap_values.apply(lambda row: self.decision_feature_names[row.argmax()], axis=1)

        return reason

    def verify_input_features(self, input_features: pd.DataFrame) -> bool:
        return set(self.input_feature_names).issubset(set(input_features.columns))

    def verify_decision_features(self, decision_features: pd.DataFrame) -> bool:
        """
        This method verifies if transform_feature method returns the features which matches with the
        decision_engine_features
        :param decision_features is the dataframe which returned by transform_feature method.
        """
        return set(self.decision_feature_names) == set(decision_features.columns)

    def predict_proba(self, feature_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        This function takes the KYC data and implements the risk engine V2
        model, and outputs both the integer score and the 'low, med, high' scores

        :param pd.DataFrame feature_df: dataframe of KYC data
        :return: pd.Series: dataframe of predicted probabilites
        """

        if not self.verify_decision_features(decision_features=feature_df):
            feature_df = self.transform_features(feature_df)

            if not self.verify_decision_features(decision_features=feature_df):
                raise LoadDefaultModelError('Decision feature error in predict_proba')

        pred_proba_np = self.model_calibration_obj.predict_proba(feature_df)

        labels_pred_ser = pd.Series(pred_proba_np[:, 1], index=feature_df.index)
        reason = self.shap_reason(feature_df)

        labels_pred_ser.rename('yPredScore')
        reason.rename('shap_reason')
        labels_pred_ser_scaled = labels_pred_ser.apply(lambda x: round(x * self.RISK_SCORE_MULTIPLIER))

        #         return labels_pred_ser_scaled
        return (labels_pred_ser_scaled, reason)

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        This function performs the overall prediction to assess recall of the model
        (where the med/high threshold is the defacto decision boundary)

        :param pd.DataFrame feature_df: input features
        :return pd.Series: series containing the prediction result (0/1)
        """

        #         prediction_df = self.predict_proba(feature_df)
        prediction_df = self.predict_proba(feature_df)[0]

        predict_df = prediction_df.apply(lambda x: 1 if x >= self.THRESHOLD_MKYC_REJECT else 0)

        return predict_df.rename('prediction')

    def predict_risk_category(self, feature_df):

        if len(self.input_feature_names) != 0:
            feature_df = feature_df[list(self.input_feature_names)]

        # output of model is sum of rows of DF (since it's a linear model)
        prediction_df = self.predict_proba(feature_df)[0]
        #         prediction_df = self.predict_proba(feature_df)

        return prediction_df.apply(self._assign_risk_category)

    def _assign_risk_category(self, risk_rating):
        """

        :param risk_rating:
        :return:
        """

        if risk_rating < self.MIN_RISK_SCORE:
            raise RuntimeError('risk score too low')
        elif risk_rating > self.MAX_RISK_SCORE:
            raise RuntimeError('risk score too high')
        elif risk_rating >= self.THRESHOLD_MKYC_REJECT:
            return 'reject'
        elif risk_rating >= self.THRESHOLD_APPROVED_MKYC:
            return 'mkyc'
        elif risk_rating < self.THRESHOLD_APPROVED_MKYC:
            return 'approved'
        else:
            raise RuntimeError('_assign_rejection_category error')

    def feature_names(self) -> FeatureNames:
        """Return a tuple of strings enumerating all feature names the model expects.
        Requires a trained model for this to work, otherwise raises
        ``XGBoostError('need to call fit or load_model first')`` .
        """
        return self.decision_feature_names

    def verify_input(self, features: EngineInputOutputT) -> None:
        """Pass silently if input data is valid, otherwise raise InvalidInputError.
                For sub-engines, currently does nothing."""

    def _decide(self, features_df: DataFrameInputT) -> Tuple[float, str]:
        """Call the ML model and then extract and return the risk score as float."""

        predict_ndarray, predict_reason = self.predict_proba(features_df)
        return predict_ndarray.to_list()[0], predict_reason.tolist()[0]

    def _reduce_feature_vector(self, feature_vector: EngineInputOutputT) -> EngineInputOutputT:
        """Return an ordered dict of features following the order in this engine's
        booster feature_names list, eliminating any non-required features.
        """

        return OrderedDict((name, feature_vector[name]) for name in self.input_feature_names)

    def decide(self, features: EngineInputOutputT, *args) -> EngineInputOutputT:

        ordered_vector = self._reduce_feature_vector(features)
        input_df = dict_to_dataframe(ordered_vector)

        if not self.verify_input_features(input_features=input_df):
            raise LoadDefaultModelError('Error with input features')

        loan_default_rating_raw, reason = self._decide(input_df)

        loan_default_engine_decision = self._assign_risk_category(loan_default_rating_raw)

        return dict(
            loan_default_rating=round_to_int(loan_default_rating_raw),
            loan_default_engine_decision=loan_default_engine_decision,
            loan_default_reason=[reason.upper() if loan_default_engine_decision == 'reject' else '']
        )


