import numpy as np
import numpy.typing as npt
from sklearn.utils import check_array
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from .core import (
    cart_bctx,
    compute_ccp_alpha_path,
    prune_tree_by_ccp_alpha,
    prune_tree_equal_siblings
)

def _set_max_depth(max_depth, n_samples):
    """Set the maximum depth of the tree."""
    if max_depth is None:
        return n_samples
    elif isinstance(max_depth, int):
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer or None.")
        return max_depth
    else:
        raise ValueError("max_depth must be a positive integer or None.")

def _set_max_features(max_features, n_features):
    """Set the maximum number of features to consider when looking for the best split."""
    if max_features is None:
        return n_features
    elif isinstance(max_features, int):
        if max_features <= 0 or max_features > n_features:
            raise ValueError("max_features must be in (0, n_features]")
        return max_features
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("max_features as a float must be in (0.0, 1.0]")
        return max(1, int(max_features * n_features))
    else:
        raise ValueError("max_features must be None, int, or float.")

def _apply(tree, X: np.ndarray) -> np.ndarray[int]:
    """Return the index of the leaf that each sample is predicted as."""
    n_samples = X.shape[0]
    leaf_indices = np.empty(n_samples, dtype=int)
    node_stack = [(0, np.arange(n_samples))]
    while len(node_stack) > 0:
        node_id, sample_indices = node_stack.pop()
        if tree['splitter'][node_id] == -1:
            leaf_indices[sample_indices] = node_id
        else:
            feature_index = tree['splitter'][node_id]
            threshold = tree['threshold'][node_id]
            left_mask = X[sample_indices, feature_index] <= threshold
            right_mask = ~left_mask
            if np.any(left_mask):
                left_indices = sample_indices[left_mask]
                node_stack.append((tree['left_child'][node_id], left_indices))
            if np.any(right_mask):
                right_indices = sample_indices[right_mask]
                node_stack.append((tree['right_child'][node_id], right_indices))
    return leaf_indices

def _explain(tree, X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    n_features = X.shape[1]
    xlbs = np.full((n_samples, n_features), -np.inf)
    xubs = np.full((n_samples, n_features), np.inf)
    node_stack = [(0, np.arange(n_samples))]
    while len(node_stack) > 0:
        node_id, sample_indices = node_stack.pop()
        if tree['splitter'][node_id] == -1:
            continue
        else:
            feature_index = tree['splitter'][node_id]
            threshold = tree['threshold'][node_id]
            left_mask = X[sample_indices, feature_index] <= threshold
            right_mask = ~left_mask
            if np.any(left_mask):
                left_indices = sample_indices[left_mask]
                xubs[left_indices, feature_index] = threshold
                node_stack.append((tree['left_child'][node_id], left_indices))
            if np.any(right_mask):
                right_indices = sample_indices[right_mask]
                xlbs[right_indices, feature_index] = threshold
                node_stack.append((tree['right_child'][node_id], right_indices))
    return xlbs, xubs

class _TreeBaseBCTX:
    """Base class for decision trees with BCTX."""
    def __init__(
            self,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_weight_fraction_leaf: float = 0.0,
            max_features: int | float | str | None = None,
            min_impurity_decrease: float = 0.0,
            min_bctx_gain: float = -1.0,
            bctx_lambda: float = 5e-3,
            bctx_mode: int = 4,
            ccp_alpha: float = 0.0,
            tau: float = -1,
            cv: int | None = 5,
            random_state: int = 42
        ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_bctx_gain = min_bctx_gain
        self.bctx_lambda = bctx_lambda
        self.bctx_mode = bctx_mode
        self.ccp_alpha = ccp_alpha
        self.cv = cv
        self.tau = tau
        self.random_state = random_state
        self.tree_ = None

    def _fit_tree(
            self, task, X, y, sample_task_weight,
            xlbs, xubs, sample_bctx_weight, bctx_lambda
        ):
        xmins = np.min(X, axis=0)
        xmaxs = np.max(X, axis=0)
        xstds = np.maximum(np.std(X, axis=0), np.finfo(float).eps)

        xlbs_used = 1 - np.isinf(xlbs).astype(int)
        xubs_used = 1 - np.isinf(xubs).astype(int)
        
        xlbs = np.maximum(xlbs, xmins)
        xubs = np.minimum(xubs, xmaxs)

        # fit base tree to get pruning path
        base_tree = cart_bctx(
            task, X, y, sample_task_weight,
            xlbs, xubs, xlbs_used, xubs_used,
            xstds, sample_bctx_weight,
            _set_max_depth(self.max_depth, X.shape[0]),
            self.min_samples_split,
            self.min_samples_leaf,
            self.min_weight_fraction_leaf,
            _set_max_features(self.max_features, X.shape[1]),
            self.min_impurity_decrease,
            self.min_bctx_gain,
            bctx_lambda,
            self.bctx_mode,
            self.random_state
        )

        if self.cv is None:
            # prune tree using given ccp_alpha
            pruned_tree = prune_tree_by_ccp_alpha(base_tree, self.ccp_alpha)
            pruned_tree = prune_tree_equal_siblings(pruned_tree, self.tau)
            return pruned_tree

        # perform cross-validation to find the best ccp_alpha
        alphas = compute_ccp_alpha_path(base_tree)
        if len(alphas) == 0:
            alphas = np.array([0.0])
        if np.min(alphas) > 0:
            alphas = np.append(0.0, alphas)  # ensure 0.0 is included

        if task == 'classif':
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
        scores = np.zeros(len(alphas))
        for trn_idx, val_idx in kf.split(X, y):
            X_trn, y_trn = X[trn_idx], y[trn_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            sample_task_weight_trn = sample_task_weight[trn_idx]
            sample_bctx_weight_trn = sample_bctx_weight[trn_idx]
            sample_task_weight_val = sample_task_weight[val_idx]
            sample_bctx_weight_val = sample_bctx_weight[val_idx]
            xlbs_trn, xubs_trn = xlbs[trn_idx], xubs[trn_idx]
            xlbs_val, xubs_val = xlbs[val_idx], xubs[val_idx]
            xlbs_used_trn, xubs_used_trn = xlbs_used[trn_idx], xubs_used[trn_idx]
            xlbs_used_val, xubs_used_val = xlbs_used[val_idx], xubs_used[val_idx]

            # fit tree on training fold
            tree_k = cart_bctx(
                task, X_trn, y_trn, sample_task_weight_trn,
                xlbs_trn, xubs_trn, xlbs_used_trn, xubs_used_trn,
                xstds, sample_bctx_weight_trn,
                _set_max_depth(self.max_depth, X_trn.shape[0]),
                self.min_samples_split,
                self.min_samples_leaf,
                self.min_weight_fraction_leaf,
                _set_max_features(self.max_features, X_trn.shape[1]),
                self.min_impurity_decrease,
                self.min_bctx_gain,
                bctx_lambda,
                self.bctx_mode,
                self.random_state
            )

            for i, ccp_alpha in enumerate(alphas):
                tree_k = prune_tree_by_ccp_alpha(tree_k, ccp_alpha)
                tree = prune_tree_equal_siblings(tree_k, self.tau)
                
                # evaluate on validation fold
                if task == 'classif':
                    y_val_pred = tree['predicted_class'][_apply(tree, X_val)]
                    task_loss = np.sum(
                        sample_task_weight_val * (y_val_pred != y_val).astype(float)
                    ) / np.maximum(np.sum(sample_task_weight_val), np.finfo(float).eps)
                else:
                    y_val_pred = tree['predicted_value'][_apply(tree, X_val)]
                    task_loss = np.sum(
                        sample_task_weight_val * (y_val_pred - y_val) ** 2
                    ) / np.maximum(np.sum(sample_task_weight_val), np.finfo(float).eps)

                bctx_loss = 0.0
                if bctx_lambda > 0.0:
                    xlbs_val_new, xubs_val_new = _explain(tree, X_val)
                    if self.bctx_mode in [1, 2]:
                        xlbs_used_val_new = 1 - np.isinf(xlbs_val_new).astype(int)
                        xubs_used_val_new = 1 - np.isinf(xubs_val_new).astype(int)
                        if self.bctx_mode == 1:
                            # JAC
                            used_val = np.bitwise_or(xlbs_used_val, xubs_used_val)
                            used_val_new = np.bitwise_or(xlbs_used_val_new, xubs_used_val_new)
                            bctx_loss = np.sum(
                                sample_bctx_weight_val * np.mean(
                                    np.abs(used_val_new - used_val), axis=1
                                )
                            ) / np.maximum(np.sum(sample_bctx_weight_val), np.finfo(float).eps)
                        if self.bctx_mode == 2:
                            # JLU
                            bctx_loss = np.sum(
                                sample_bctx_weight_val * np.mean(
                                    np.abs(xlbs_used_val_new - xlbs_used_val) +
                                    np.abs(xubs_used_val_new - xubs_used_val),
                                    axis=1
                                ) / 2
                            ) / np.maximum(np.sum(sample_bctx_weight_val), np.finfo(float).eps)
                    else:
                        xlbs_val_new = np.maximum(xlbs_val_new, xmins)
                        xubs_val_new = np.minimum(xubs_val_new, xmaxs)
                        if self.bctx_mode == 3:
                            # MAE
                            bctx_loss = np.sum(
                                sample_bctx_weight_val * np.mean(
                                    np.abs((xlbs_val_new - xlbs_val) / xstds) +
                                    np.abs((xubs_val_new - xubs_val) / xstds),
                                    axis=1
                                ) / 2
                            ) / np.maximum(np.sum(sample_bctx_weight_val), np.finfo(float).eps)
                        if self.bctx_mode == 4:
                            # MSE
                            bctx_loss = np.sum(
                                sample_bctx_weight_val * np.mean(
                                    ((xlbs_val_new - xlbs_val) / xstds) ** 2 +
                                    ((xubs_val_new - xubs_val) / xstds) ** 2,
                                    axis=1
                                ) / 2
                            ) / np.maximum(np.sum(sample_bctx_weight_val), np.finfo(float).eps)
                
                # combine classification and BCTX losses
                total_loss = task_loss + bctx_lambda * bctx_loss
                scores[i] += total_loss

        # select the best alpha and prune the tree
        best_alpha = alphas[np.argmin(scores)]
        pruned_tree = prune_tree_by_ccp_alpha(base_tree, best_alpha)
        pruned_tree = prune_tree_equal_siblings(pruned_tree, self.tau)
        return pruned_tree

    def apply(self, X: npt.ArrayLike) -> np.ndarray[int]:
        """Return the index of the leaf that each sample is predicted as.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_leaves : ndarray of shape (n_samples,)
            For each sample in X, return the index of the leaf it is predicted as.
        """
        X = check_array(X)
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        return _apply(self.tree_, X)
    
    def explain(self, X: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the decision rule as intervals for each feature for each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        xlbs : ndarray of shape (n_samples, n_features)
            The lower bounds for each feature for each sample.
        xubs : ndarray of shape (n_samples, n_features)
            The upper bounds for each feature for each sample.
        """
        X = check_array(X)
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        return _explain(self.tree_, X)

    def num_leaves(self) -> int:
        """Return the number of leaves in the fitted tree.
        Returns
        -------
        n_leaves : int
            The number of leaves in the tree.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        return np.sum(self.tree_['splitter'] == -1).item()
    
    def cost_complexity_pruning_path(self) -> npt.ArrayLike:
        """Compute the cost-complexity pruning path.
        Returns
        -------
        ccp_alphas : ndarray of shape (n_pruned_trees,)
            The effective alphas of the pruned trees.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        return compute_ccp_alpha_path(self.tree_)
    
    def prune_tree(self, ccp_alpha: float) -> None:
        """Prune the tree using cost-complexity pruning.
        Parameters
        ----------
        ccp_alpha : float
            The complexity parameter used for pruning.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        self.tree_ = prune_tree_by_ccp_alpha(self.tree_, ccp_alpha)
    
    def plot_2d_splits(
            self, i: int, j: int,
            vmin: npt.ArrayLike,
            vmax: npt.ArrayLike,
            line_color: str = 'k',
            line_width: float = 1.0,
            line_style: str = 'solid',
            ax: plt.Axes | None = None
        ) -> plt.Axes:
        """Plot the 2D splits for features i and j.
        Parameters
        ----------
        i : int
            The index of the first feature.
        j : int
            The index of the second feature.
        vmin : npt.ArrayLike
            The minimum values for the two features.
        vmax : npt.ArrayLike
            The maximum values for the two features.
        ax : plt.Axes, optional
            The axes to plot on. If None, a new figure and axes are created.

        Returns
        -------
        ax : plt.Axes
            The axes on which the plot was drawn.
        """
        if self.tree_ is None:
            raise ValueError("The decision tree is not fitted yet.")
        if ax is None:
            _, ax = plt.subplots()

        def plot_node(node_id: int, node_lb: list[float], node_ub: list[float]):
            feature = self.tree_["splitter"][node_id]
            if feature == -1:  # leaf node
                return
            
            axis = -1
            if feature == i: axis = 0
            elif feature == j: axis = 1

            if axis == -1:
                plot_node(self.tree_["left_child"][node_id], node_lb, node_ub)
                plot_node(self.tree_["right_child"][node_id], node_lb, node_ub)
            else:
                threshold = self.tree_["threshold"][node_id]
                if axis == 0:
                    ax.vlines(
                        threshold, node_lb[1], node_ub[1], colors=line_color,
                        linewidth=line_width, linestyles=line_style
                    )
                else:
                    ax.hlines(
                        threshold, node_lb[0], node_ub[0], colors=line_color,
                        linewidth=line_width, linestyles=line_style
                    )
                tmp = node_ub[axis]
                node_ub[axis] = threshold
                plot_node(self.tree_["left_child"][node_id], node_lb, node_ub)
                node_ub[axis] = tmp

                tmp = node_lb[axis]
                node_lb[axis] = threshold
                plot_node(self.tree_["right_child"][node_id], node_lb, node_ub)
                node_lb[axis] = tmp

        plot_node(0, [vmin[0], vmin[1]], [vmax[0], vmax[1]])
        ax.set_xlim([vmin[0], vmax[0]])
        ax.set_ylim([vmin[1], vmax[1]])
        return ax