from sklearn.linear_model import LinearRegression, LogisticRegression
from causallib.estimation import Standardization, StratifiedStandardization, IPW, AIPW
from econml.dr import DRLearner

from causal_estimators.base import BaseCausallibIteEstimator, BaseEconMLEstimator


STR_TO_DOUBLY_ROBUST = {
    'aipw': AIPW,
    'vanilla': AIPW,  # Use AIPW as the default doubly robust estimator
}
DOUBLY_ROBUST_TYPES = STR_TO_DOUBLY_ROBUST.keys()

STR_TO_STANDARDIZATION = {
    'standardization': Standardization,
    'stratified_standardization': StratifiedStandardization,
    'stratifiedstandardization': StratifiedStandardization,
    'stratified': StratifiedStandardization,
}

TRIM_EPS = 0.01


class DoublyRobustEstimator(BaseCausallibIteEstimator):

    def __init__(self, outcome_model=LinearRegression(),
                 prop_score_model=LogisticRegression(),
                 doubly_robust_type='vanilla',
                 standardization_type='standardization',
                 trim_weights=False, trim_eps=None, stabilized=False):

        if doubly_robust_type not in DOUBLY_ROBUST_TYPES:
            raise ValueError('Invalid double_robust_type. Valid types: {}'
                             .format(list(DOUBLY_ROBUST_TYPES)))
        if standardization_type not in STR_TO_STANDARDIZATION.keys():
            raise ValueError('Invalid standardization_type. Valid types: {}'
                             .format(list(STR_TO_STANDARDIZATION.keys())))

        if trim_weights and trim_eps is None:
            trim_eps = TRIM_EPS

        # AIPW (Augmented Inverse Probability Weighting) combines both outcome and propensity models
        doubly_robust = AIPW(
            outcome_model=outcome_model, 
            weight_model=prop_score_model
        )

        super().__init__(causallib_estimator=doubly_robust)


class DoublyRobustLearner(BaseEconMLEstimator):

    def __init__(self, outcome_model=LinearRegression(),
                 prop_score_model=LogisticRegression(),
                 final_model=LinearRegression(), trim_eps=1e-6,
                 random_state=None):
        # TODO: add other options that DRLearner allows?
        drlearner = DRLearner(model_propensity=prop_score_model,
                              model_regression=outcome_model,
                              model_final=final_model, min_propensity=trim_eps,
                              random_state=random_state,
                              cv=1)
        super().__init__(drlearner)
