import time
import numpy as np
import numpy.typing as npt
import pandas as pd
import graphviz
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import check_X_y, compute_class_weight
from ._base import _TreeBaseBCTX


__BCTX_MODE__ = {
    'JAC': 1,
    'JLU': 2,
    'MAE': 3,
    'MSE': 4
}

class DecisionTreeClassifierBCTX(_TreeBaseBCTX, BaseEstimator, ClassifierMixin):
    """CART-BCTX classifier.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than specified samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
    max_features : int, float, string or None, default=None
        The number of features to consider when looking for the best split.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of
        the impurity (including BCTX loss) greater than this value.
    min_bctx_gain : float, default=-1e9
        The minimum BCTX gain required to split a node.
        Note that a node will be finally split according to min_impurity_decrease.
    bctx_lambda : float, default=5e-3
        The regularization parameter for BCTX.
    bctx_mode : str, default='MSE'
        The mode of BCTX. Must be one of 'JAC', 'JLU', 'MAE', 'MSE'.
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes.
    cv : int or None, default=5
        Number of cross-validation folds for Cost-Complexity Pruning.
    random_state : int, default=42
        Controls the randomness of the estimator.
    """
    def __init__(
            self,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: int | float | str | None = None,
            min_impurity_decrease: float = 0.0,
            min_bctx_gain: float = -1e9,
            bctx_lambda: float = 5e-3,
            bctx_mode: str = 'MSE',
            ccp_alpha: float = 0.0,
            class_weight: None | str | dict = None,
            cv: int | None = 5,
            random_state: int = 42
        ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_bctx_gain=min_bctx_gain,
            bctx_lambda=bctx_lambda,
            bctx_mode=__BCTX_MODE__[bctx_mode],
            ccp_alpha=ccp_alpha,
            tau=-1,
            cv=cv,
            random_state=random_state
        )
        self.class_weight = class_weight

    def _set_sample_clf_weight(self, X, y, sample_clf_weight):
        if sample_clf_weight is None:
            if self.class_weight is None:
                sample_clf_weight = np.ones(X.shape[0], dtype=float)
            elif self.class_weight == 'balanced':
                class_weights = compute_class_weight('balanced', classes=self.classes_, y=y)
                sample_clf_weight = class_weights[y]
            elif isinstance(self.class_weight, dict):
                class_weights = np.array([self.class_weight[cls] for cls in self.classes_])
                sample_clf_weight = class_weights[y]
            else:
                raise ValueError("class_weight must be None, 'balanced', or a dictionary.")
        else:
            sample_clf_weight = np.asarray(sample_clf_weight, dtype=float)
            if sample_clf_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_clf_weight must have the same length as X and y.")
        return sample_clf_weight
    
    def fit(
            self,
            X: npt.ArrayLike,
            y: npt.ArrayLike,
            sample_clf_weight: npt.ArrayLike | None = None,
            feature_names: npt.ArrayLike | None = None
        ) -> "DecisionTreeClassifierBCTX":
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_clf_weight : array-like of shape (n_samples,), default=None
            Sample weights for classification loss.
            If None, then samples are equally weighted.
        feature_names : array-like of shape (n_features,), default=None
            Names of the features. If None and X is a pandas DataFrame,
            the column names of the DataFrame are used.

        Returns
        -------
        self : DecisionTreeClassifierBCTX
            Fitted estimator.
        """
        # check input data
        if feature_names is not None:
            feature_names = np.asarray(feature_names)
        elif isinstance(X, pd.DataFrame):
            feature_names = X.columns.to_numpy()

        X, y = check_X_y(X, y)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)

        if feature_names is None:
            feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X.shape[1]

        # set sample weight
        sample_clf_weight = self._set_sample_clf_weight(X, y, sample_clf_weight)

        # build tree without BCTX
        start_time = time.perf_counter()
        self.tree_ = self._fit_tree(
            'classif', X, y, sample_clf_weight,
            np.zeros_like(X), np.zeros_like(X), np.zeros(X.shape[0]), 0.0 # dummy
        )
        self.fit_time_ = time.perf_counter() - start_time
        return self

    def update(
            self,
            X: npt.ArrayLike,
            y: npt.ArrayLike,
            sample_clf_weight: npt.ArrayLike | None = None,
            sample_bctx_weight: npt.ArrayLike | None = None
        ) -> "DecisionTreeClassifierBCTX":
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_clf_weight : array-like of shape (n_samples,), default=None
            Sample weights for classification loss.
            If None, then samples are equally weighted.
        sample_bctx_weight : array-like of shape (n_samples,), default=None
            Sample weights for BCTX loss.
            If None, then samples are equally weighted.

        Returns
        -------
        self : DecisionTreeClassifierBCTX
            Fitted estimator.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet. Call fit() before update().")
        
        # check input data
        X, y = check_X_y(X, y)
        y = self.label_encoder_.transform(y)

        # set sample weight
        sample_clf_weight = self._set_sample_clf_weight(X, y, sample_clf_weight)
        if sample_bctx_weight is None:
            y_pred = self.label_encoder_.transform(self.predict(X))
            sample_bctx_weight = (y_pred == y).astype(float)
        else:
            sample_bctx_weight = np.asarray(sample_bctx_weight, dtype=float)
            if sample_bctx_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_bctx_weight must have the same length as X and y.")
            
        # build tree with BCTX
        start_time = time.perf_counter()
        xlbs, xubs = self.explain(X)
        self.tree_ = self._fit_tree(
            'classif', X, y, sample_clf_weight,
            xlbs, xubs, sample_bctx_weight, self.bctx_lambda
        )
        self.fit_time_ = time.perf_counter() - start_time
        return self
    
    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict class for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.
        """
        leaf_indices = self.apply(X)
        y_pred = self.tree_['predicted_class'][leaf_indices]
        y_pred = self.label_encoder_.inverse_transform(y_pred)
        return y_pred
    
    def export_graphviz(
            self,
            report_n_samples: bool = True,
            report_class_distribution: bool = True,
            round_features: int | None = None
        ) -> graphviz.Digraph:
        """Export the decision tree in Graphviz format.

        Parameters
        ----------
        report_n_samples : bool, default=True
            Whether to report the number of samples at each node.
        report_class_distribution : bool, default=True
            Whether to report the class distribution at each node.
        round_features : int | None, default=None
            Number of decimal places to round the reported feature values.
            If None, no rounding is performed.

        Returns
        -------
        dot : graphviz.Digraph
            The decision tree in Graphviz format.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        
        dot = graphviz.Digraph()

        def proc_node_recur(node_id):
            feature = self.tree_["splitter"][node_id]
            if feature == -1:  # leaf node
                proba = self.tree_["class_probs"][node_id]
                n_samples = self.tree_["n_samples"][node_id]
                prediction = self.tree_["predicted_class"][node_id]
                label = f"class: {self.classes_[prediction]}\n"
                if report_n_samples:
                    label += f"samples = {n_samples}\n"
                if report_class_distribution:
                    label += "\n".join([
                        f"class {cls}: {prb:.2f}"
                        for cls, prb in zip(self.classes_, proba)
                    ])
                dot.node(
                    str(node_id), label=label,
                    shape='box', style='filled', fillcolor='lightgrey'
                )
            else: # internal node
                threshold = self.tree_["threshold"][node_id]
                if round_features is not None:
                    threshold = round(threshold, round_features)
                n_samples = self.tree_["n_samples"][node_id]
                proba = self.tree_["class_probs"][node_id]
                label = f"{self.feature_names_in_[feature]} <= {threshold}\n"
                if report_n_samples:
                    label += f"samples = {n_samples}\n"
                if report_class_distribution:
                    label += "\n".join([
                        f"class {cls}: {prb:.2f}"
                        for cls, prb in zip(self.classes_, proba)
                    ])
                dot.node(str(node_id), label=label)
                left_child = self.tree_["left_child"][node_id]
                right_child = self.tree_["right_child"][node_id]
                proc_node_recur(left_child)
                proc_node_recur(right_child)
                dot.edge(str(node_id), str(left_child), label="Yes")
                dot.edge(str(node_id), str(right_child), label="No")

        proc_node_recur(0)
        return dot
    
    @property
    def classes_(self) -> np.ndarray:
        """Return the classes seen during fit.
        Returns
        -------
        classes_ : ndarray of shape (n_classes,)
            The classes seen during fit.
        """
        if not hasattr(self, 'label_encoder_'):
            raise ValueError("The decision tree classifier is not fitted yet.")
        return self.label_encoder_.classes_
    
    @property
    def n_classes_(self) -> int:
        """Return the number of classes seen during fit.
        Returns
        -------
        n_classes_ : int
            The number of classes seen during fit.
        """
        if not hasattr(self, 'label_encoder_'):
            raise ValueError("The decision tree classifier is not fitted yet.")
        return len(self.label_encoder_.classes_)
    
class DecisionTreeRegressorBCTX(_TreeBaseBCTX, BaseEstimator, RegressorMixin):
    """CART-BCTX regressor.
    
    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than specified samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
    max_features : int, float, string or None, default=None
        The number of features to consider when looking for the best split.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of
        the impurity (including BCTX loss) greater than this value.
    min_bctx_gain : float, default=-1e9
        The minimum BCTX gain required to split a node.
        Note that a node will be finally split according to min_impurity_decrease.
    bctx_lambda : float, default=5e-3
        The regularization parameter for BCTX.
    bctx_mode : str, default='MSE'
        The mode of BCTX. Must be one of 'JAC', 'JLU', 'MAE', 'MSE'.
    bc_epsilon : float, default=0.1
        The regression absolute error threshold for
        determining whether a sample is correctly predicted.
        This is only used for calculating BCTX weights
        in `update` function when sample_bctx_weight is None.
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    cv : int or None, default=5
        Number of cross-validation folds for Cost-Complexity Pruning.
    random_state : int, default=42
        Controls the randomness of the estimator.
    """
    def __init__(
            self,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: int | float | str | None = None,
            min_impurity_decrease: float = 0.0,
            min_bctx_gain: float = -1e9,
            bctx_lambda: float = 5e-3,
            bctx_mode: str = 'MSE',
            bc_epsilon: float = 0.1,
            ccp_alpha: float = 0.0,
            tau: float = 1e-3,
            cv: int | None = 5,
            random_state: int = 42
        ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_bctx_gain=min_bctx_gain,
            bctx_lambda=bctx_lambda,
            bctx_mode=__BCTX_MODE__[bctx_mode],
            ccp_alpha=ccp_alpha,
            tau=tau,
            cv=cv,
            random_state=random_state
        )
        self.bc_epsilon = bc_epsilon

    def fit(
            self,
            X: npt.ArrayLike,
            y: npt.ArrayLike,
            sample_reg_weight: npt.ArrayLike | None = None,
            feature_names: npt.ArrayLike | None = None
        ) -> "DecisionTreeRegressorBCTX":
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_reg_weight : array-like of shape (n_samples,), default=None
            Sample weights for regression loss.
            If None, then samples are equally weighted.
        feature_names : array-like of shape (n_features,), default=None
            Names of the features. If None and X is a pandas DataFrame,
            the column names of the DataFrame are used.

        Returns
        -------
        self : DecisionTreeRegressorBCTX
            Fitted estimator.
        """
        # check input data
        if feature_names is not None:
            feature_names = np.asarray(feature_names)
        elif isinstance(X, pd.DataFrame):
            feature_names = X.columns.to_numpy()

        X, y = check_X_y(X, y)
        self.scaler_ = MinMaxScaler()
        y = self.scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        if feature_names is None:
            feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X.shape[1]

        # set sample weight
        if sample_reg_weight is None:
            sample_reg_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_reg_weight = np.asarray(sample_reg_weight, dtype=float)
            if sample_reg_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_reg_weight must have the same length as X and y.")

        # build tree without BCTX
        start_time = time.perf_counter()
        self.tree_ = self._fit_tree(
            'regress', X, y, sample_reg_weight,
            np.zeros_like(X), np.zeros_like(X), np.zeros(X.shape[0]), 0.0 # dummy
        )
        self.fit_time_ = time.perf_counter() - start_time
        return self

    def update(
            self,
            X: npt.ArrayLike,
            y: npt.ArrayLike,
            sample_reg_weight: npt.ArrayLike | None = None,
            sample_bctx_weight: npt.ArrayLike | None = None
        ) -> "DecisionTreeRegressorBCTX":
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_reg_weight : array-like of shape (n_samples,), default=None
            Sample weights for regression loss.
            If None, then samples are equally weighted.
        sample_bctx_weight : array-like of shape (n_samples,), default=None
            Sample weights for BCTX loss.
            If None, then samples are equally weighted.

        Returns
        -------
        self : DecisionTreeRegressorBCTX
            Fitted estimator.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet. Call fit() before update().")
        
        # check input data
        X, y = check_X_y(X, y)
        y = self.scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        # set sample weight
        if sample_reg_weight is None:
            sample_reg_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_reg_weight = np.asarray(sample_reg_weight, dtype=float)
            if sample_reg_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_reg_weight must have the same length as X and y.")
        
        if sample_bctx_weight is None:
            y_pred = self.predict(X)
            y_orig = self.scaler_.inverse_transform(y.reshape(-1, 1)).flatten()
            sample_bctx_weight = (np.abs(y_pred - y_orig) <= self.bc_epsilon).astype(float)
        else:
            sample_bctx_weight = np.asarray(sample_bctx_weight, dtype=float)
            if sample_bctx_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_bctx_weight must have the same length as X and y.")
            
        # build tree with BCTX
        start_time = time.perf_counter()
        xlbs, xubs = self.explain(X)
        self.tree_ = self._fit_tree(
            'regress', X, y, sample_reg_weight,
            xlbs, xubs, sample_bctx_weight, self.bctx_lambda
        )
        self.fit_time_ = time.perf_counter() - start_time
        return self
    
    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict regression target for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        leaf_indices = self.apply(X)
        y_pred = self.tree_['predicted_value'][leaf_indices]
        y_pred = self.scaler_.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
    
    def export_graphviz(
            self,
            report_n_samples: bool = True,
            round_features: int | None = None,
            round_predicts: int | None = None
        ) -> graphviz.Digraph:
        """Export the decision tree in Graphviz format.

        Parameters
        ----------
        report_n_samples : bool, default=True
            Whether to report the number of samples at each node.
        round_features : int | None, default=None
            Number of decimal places to round the reported feature values.
            If None, no rounding is performed.
        round_predicts : int | None, default=None
            Number of decimal places to round the predicted values at leaf nodes.
            If None, no rounding is performed.

        Returns
        -------
        dot : graphviz.Digraph
            The decision tree in Graphviz format.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        
        dot = graphviz.Digraph()

        def proc_node_recur(node_id):
            feature = self.tree_["splitter"][node_id]
            if feature == -1:  # leaf node
                value = self.tree_["predicted_value"][node_id]
                value = self.scaler_.inverse_transform([[value]]).flatten()[0]
                n_samples = self.tree_["n_samples"][node_id]
                if round_predicts is not None:
                    value = round(value, round_predicts)
                label = f"value = {value}\n"
                if report_n_samples:
                    label += f"samples = {n_samples}\n"
                dot.node(
                    str(node_id), label=label,
                    shape='box', style='filled', fillcolor='lightgrey'
                )
            else: # internal node
                threshold = self.tree_["threshold"][node_id]
                if round_features is not None:
                    threshold = round(threshold, round_features)
                n_samples = self.tree_["n_samples"][node_id]
                label = f"{self.feature_names_in_[feature]} <= {threshold}\n"
                if report_n_samples:
                    label += f"samples = {n_samples}\n"
                dot.node(str(node_id), label=label)
                left_child = self.tree_["left_child"][node_id]
                right_child = self.tree_["right_child"][node_id]
                proc_node_recur(left_child)
                proc_node_recur(right_child)
                dot.edge(str(node_id), str(left_child), label="Yes")
                dot.edge(str(node_id), str(right_child), label="No")

        proc_node_recur(0)
        return dot