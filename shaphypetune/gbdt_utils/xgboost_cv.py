from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from xgboost.sklearn import (
    _wrap_evaluation_matrices, 
    _objective_decorator,
    _metric_decorator,
    ltr_metric_decorator,
    TrainingCallback,
    _deprecate_positional_args
)
from xgboost.config import config_context
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from xgboost.core import ArrayLike, Booster, IterationRange
from xgboost.compat import XGBStratifiedKFold
import warnings


class XGBClassifierCV(XGBClassifier):
    """XGBoost Classifier with Cross-Validation functionality.
    
    This class inherits from XGBClassifier but uses xgboost.cv() instead of 
    xgboost.train() in the fit method to perform cross-validation during training.
    
    Parameters
    ----------
    cv_nfold : int, default=5
        Number of cross-validation folds.
    cv_nfolds : XGBStratifiedKFold, default=None
        a KFold or StratifiedKFold instance or list of fold indices
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n`` th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n`` th fold.
    cv_stratified : bool, default=True
        Whether to use stratified cross-validation.
    cv_shuffle : bool, default=True
        Whether to shuffle data before cross-validation.
    cv_seed : int, default=0
        Random seed for cross-validation.
    cv_as_pandas : bool, default=True
        Return cv results as pandas DataFrame.
    cv_verbose_eval : bool or int, default=False
        Whether to display progress during CV.
    cv_show_stdv : bool, default=True
        Whether to display standard deviation in cv results.
    
    All other parameters are inherited from XGBClassifier.
    """
    
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        cv_nfold: int = 5,
        cv_folds: XGBStratifiedKFold = None,
        cv_stratified: bool = True,
        cv_shuffle: bool = True,
        cv_seed: int = 0,
        cv_as_pandas: bool = True,
        cv_verbose_eval: Optional[Union[bool, int]] = None,
        cv_show_stdv: bool = True,
        **kwargs: Any
    ) -> None:
        # Initialize parent class
        super().__init__(**kwargs)
        
        # CV-specific parameters
        self.cv_nfold = cv_nfold
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        self.cv_shuffle = cv_shuffle
        self.cv_seed = cv_seed
        self.cv_as_pandas = cv_as_pandas
        self.cv_verbose_eval = cv_verbose_eval
        self.cv_show_stdv = cv_show_stdv
        
        # CV results storage
        self.cv_results_ = None
        self.cv_best_iteration_ = None

    
    def get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost-specific parameters, excluding CV parameters.
        
        This ensures CV parameters don't interfere with XGBoost training.
        """
        # Get all parameters from parent class
        params = super().get_xgb_params()
        
        # Filter out cv_* parameters that might have leaked in
        filtered_params = {
            key: value for key, value in params.items() 
            if not key.startswith('cv_')
        }
        
        return filtered_params
    
    
    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, "XGBClassifier"]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBClassifierCV":
        """Fit gradient boosting classifier with cross-validation.
        
        This method performs cross-validation using xgboost.cv() and then trains
        a final model on the full dataset using the optimal number of boosting rounds.
        """
        with config_context(verbosity=self.verbosity):
            # Prepare classes and validation similar to parent class
            from xgboost.data import _is_polars_lazyframe, _is_cudf_df, _is_cudf_ser, _is_cupy_alike
            from xgboost.compat import import_cupy
            
            if _is_polars_lazyframe(y):
                y = y.collect()
            if _is_cudf_df(y) or _is_cudf_ser(y):
                cp = import_cupy()
                classes = cp.unique(y.values)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            elif _is_cupy_alike(y):
                cp = import_cupy()
                classes = cp.unique(y)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            else:
                classes = np.unique(np.asarray(y))
                self.n_classes_ = len(classes)
                expected_classes = self.classes_
                
            if (
                classes.shape != expected_classes.shape
                or not (classes == expected_classes).all()
            ):
                raise ValueError(
                    f"Invalid classes inferred from unique values of `y`.  "
                    f"Expected: {expected_classes}, got {classes}"
                )

            params = self.get_xgb_params()
            
            if callable(self.objective):
                obj = _objective_decorator(self.objective)
                params["objective"] = "binary:logistic"
            else:
                obj = None

            if self.n_classes_ > 2:
                if params.get("objective", None) != "multi:softmax":
                    params["objective"] = "multi:softprob"
                params["num_class"] = self.n_classes_

            model, metric, params, feature_weights = self._configure_fit(
                xgb_model, params, feature_weights
            )
            
            # Create training DMatrix
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=self.missing,
                X=X,
                y=y,
                group=None,
                qid=None,
                sample_weight=sample_weight,
                base_margin=base_margin,
                feature_weights=feature_weights,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                eval_group=None,
                eval_qid=None,
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=self.feature_types,
            )

            # Prepare metrics for CV
            cv_metrics = []
            if self.eval_metric is not None:
                if isinstance(self.eval_metric, list):
                    cv_metrics = self.eval_metric
                elif isinstance(self.eval_metric, str):
                    cv_metrics = [self.eval_metric]
                elif callable(self.eval_metric):
                    # For callable metrics, we'll handle them differently
                    pass

            # Perform cross-validation using the official cv function
            cv_results = xgb.cv(
                params=params,
                dtrain=train_dmatrix,
                num_boost_round=self.get_num_boosting_rounds(),
                nfold=self.cv_nfold,
                stratified=self.cv_stratified,
                folds=self.cv_folds,
                metrics=cv_metrics,  # Use metrics parameter instead of eval_metric in params
                obj=obj,
                maximize=None,
                early_stopping_rounds=self.early_stopping_rounds,
                fpreproc=None,
                as_pandas=self.cv_as_pandas,
                verbose_eval=self.cv_verbose_eval,
                show_stdv=self.cv_show_stdv,
                seed=self.cv_seed,
                callbacks=self.callbacks,
                shuffle=self.cv_shuffle,
                custom_metric=metric, # Custom evaluation function
            )
            
            # Store CV results
            self.cv_results_ = cv_results
                
            # Determine optimal number of boosting rounds
            optimal_rounds = self.get_num_boosting_rounds()
            if self.early_stopping_rounds is not None:
                if self.cv_as_pandas and hasattr(cv_results, 'shape') and len(cv_results) > 0:
                    # Find the best iteration from CV results
                    # Look for test metrics (validation metrics)
                    test_metric_cols = [col for col in cv_results.columns if 'test-' in col and '-mean' in col]
                    if test_metric_cols:
                        best_metric_col = test_metric_cols[0]  # Use first test metric
                        # Determine if we should minimize or maximize
                        if any(metric_name in best_metric_col.lower() for metric_name in ['error', 'logloss', 'rmse', 'mae']):
                            best_iteration = cv_results[best_metric_col].idxmin()
                        else:
                            best_iteration = cv_results[best_metric_col].idxmax()
                        optimal_rounds = best_iteration + 1
                elif not self.cv_as_pandas and isinstance(cv_results, dict):
                    # Handle dict format
                    for key in cv_results.keys():
                        if 'test-' in key and '-mean' in key:
                            if any(metric_name in key.lower() for metric_name in ['error', 'logloss', 'rmse', 'mae']):
                                best_iteration = np.argmin(cv_results[key])
                            else:
                                best_iteration = np.argmax(cv_results[key])
                            optimal_rounds = best_iteration + 1
                            break

            # Train final model on full data with optimal rounds
            evals_result: TrainingCallback.EvalsLog = {}
            self.cv_best_iteration_ = optimal_rounds
            self._Booster = xgb.train(
                params,
                train_dmatrix,
                optimal_rounds,
                evals=evals,
                early_stopping_rounds=None,  # Don't use early stopping for final model
                evals_result=evals_result,
                obj=obj,
                custom_metric=metric,
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=None,  # Don't use callbacks for final training
            )

            if not callable(self.objective):
                self.objective = params["objective"]

            self._set_evaluation_result(evals_result)
            return self

    def get_cv_results(self) -> Any:
        """Get cross-validation results.
        
        Returns
        -------
        cv_results : pandas.DataFrame or dict
            Cross-validation results. Format depends on cv_as_pandas parameter.
        """
        if self.cv_results_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.cv_results_


class XGBRegressorCV(XGBRegressor):
    """XGBoost Regressor with Cross-Validation functionality.
    
    This class inherits from XGBRegressor but uses xgboost.cv() instead of 
    xgboost.train() in the fit method to perform cross-validation during training.
    
    Parameters
    ----------
    cv_nfold : int, default=5
        Number of cross-validation folds.
    cv_nfolds : XGBStratifiedKFold, default=None
        a KFold or StratifiedKFold instance or list of fold indices
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n`` th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n`` th fold.
    cv_stratified : bool, default=False
        Whether to use stratified cross-validation (typically False for regression).
    cv_shuffle : bool, default=True
        Whether to shuffle data before cross-validation.
    cv_seed : int, default=0
        Random seed for cross-validation.
    cv_as_pandas : bool, default=True
        Return cv results as pandas DataFrame.
    cv_verbose_eval : bool or int, default=False
        Whether to display progress during CV.
    cv_show_stdv : bool, default=True
        Whether to display standard deviation in cv results.
    
    All other parameters are inherited from XGBRegressor.
    """
    
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        cv_nfold: int = 5,
        cv_folds: XGBStratifiedKFold = None,
        cv_stratified: bool = False,  # Typically False for regression
        cv_shuffle: bool = True,
        cv_seed: int = 0,
        cv_as_pandas: bool = True,
        cv_verbose_eval: Optional[Union[bool, int]] = None,
        cv_show_stdv: bool = True,
        **kwargs: Any
    ) -> None:
        # Initialize parent class
        super().__init__(**kwargs)
        
        # CV-specific parameters
        self.cv_nfold = cv_nfold
        self.cv_folds = cv_folds
        self.cv_stratified = cv_stratified
        self.cv_shuffle = cv_shuffle
        self.cv_seed = cv_seed
        self.cv_as_pandas = cv_as_pandas
        self.cv_verbose_eval = cv_verbose_eval
        self.cv_show_stdv = cv_show_stdv
        
        # CV results storage
        self.cv_results_ = None
        self.cv_best_iteration_ = None

    
    def get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost-specific parameters, excluding CV parameters.
        
        This ensures CV parameters don't interfere with XGBoost training.
        """
        # Get all parameters from parent class
        params = super().get_xgb_params()
        
        # Filter out cv_* parameters that might have leaked in
        filtered_params = {
            key: value for key, value in params.items() 
            if not key.startswith('cv_')
        }
        
        return filtered_params
    

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, "XGBRegressor"]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBRegressorCV":
        """Fit gradient boosting regressor with cross-validation.
        
        This method performs cross-validation using xgboost.cv() and then trains
        a final model on the full dataset using the optimal number of boosting rounds.
        """
        with config_context(verbosity=self.verbosity):
            params = self.get_xgb_params()
            
            if callable(self.objective):
                obj = _objective_decorator(self.objective)
                params["objective"] = "reg:squarederror"
            else:
                obj = None

            model, metric, params, feature_weights = self._configure_fit(
                xgb_model, params, feature_weights
            )
            
            # Create training DMatrix
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=self.missing,
                X=X,
                y=y,
                group=None,
                qid=None,
                sample_weight=sample_weight,
                base_margin=base_margin,
                feature_weights=feature_weights,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                eval_group=None,
                eval_qid=None,
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=self.feature_types,
            )

            # Prepare metrics for CV
            cv_metrics = []
            if self.eval_metric is not None:
                if isinstance(self.eval_metric, list):
                    cv_metrics = self.eval_metric
                elif isinstance(self.eval_metric, str):
                    cv_metrics = [self.eval_metric]
                elif callable(self.eval_metric):
                    # For callable metrics, we'll handle them differently
                    pass

            # Perform cross-validation using the official cv function
            cv_results = xgb.cv(
                params=params,
                dtrain=train_dmatrix,
                num_boost_round=self.get_num_boosting_rounds(),
                nfold=self.cv_nfold,
                stratified=self.cv_stratified,
                folds=self.cv_folds,
                metrics=cv_metrics,  # Use metrics parameter instead of eval_metric in params
                obj=obj,
                maximize=None,
                early_stopping_rounds=self.early_stopping_rounds,
                fpreproc=None,
                as_pandas=self.cv_as_pandas,
                verbose_eval=self.cv_verbose_eval,
                show_stdv=self.cv_show_stdv,
                seed=self.cv_seed,
                callbacks=self.callbacks,
                shuffle=self.cv_shuffle,
                custom_metric=metric,  # Custom evaluation function
            )
            
            # Store CV results
            self.cv_results_ = cv_results
                
            # Determine optimal number of boosting rounds
            optimal_rounds = self.get_num_boosting_rounds()
            if self.early_stopping_rounds is not None:
                if self.cv_as_pandas and hasattr(cv_results, 'shape') and len(cv_results) > 0:
                    # Find the best iteration from CV results
                    # Look for test metrics (validation metrics)
                    test_metric_cols = [col for col in cv_results.columns if 'test-' in col and '-mean' in col]
                    if test_metric_cols:
                        best_metric_col = test_metric_cols[0]  # Use first test metric
                        # Determine if we should minimize or maximize
                        if any(metric_name in best_metric_col.lower() for metric_name in ['rmse', 'mae', 'error']):
                            best_iteration = cv_results[best_metric_col].idxmin()
                        else:
                            best_iteration = cv_results[best_metric_col].idxmax()
                        optimal_rounds = best_iteration + 1
                elif not self.cv_as_pandas and isinstance(cv_results, dict):
                    # Handle dict format
                    for key in cv_results.keys():
                        if 'test-' in key and '-mean' in key:
                            if any(metric_name in key.lower() for metric_name in ['rmse', 'mae', 'error']):
                                best_iteration = np.argmin(cv_results[key])
                            else:
                                best_iteration = np.argmax(cv_results[key])
                            optimal_rounds = best_iteration + 1
                            break

            # Train final model on full data with optimal rounds
            evals_result: TrainingCallback.EvalsLog = {}
            self.cv_best_iteration_ = optimal_rounds
            self._Booster = xgb.train(
                params,
                train_dmatrix,
                optimal_rounds,
                evals=evals,
                early_stopping_rounds=None,  # Don't use early stopping for final model
                evals_result=evals_result,
                obj=obj,
                custom_metric=metric,
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=None,  # Don't use callbacks for final training
            )

            self._set_evaluation_result(evals_result)
            return self

    def get_cv_results(self) -> Any:
        """Get cross-validation results.
        
        Returns
        -------
        cv_results : pandas.DataFrame or dict
            Cross-validation results. Format depends on cv_as_pandas parameter.
        """
        if self.cv_results_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.cv_results_