# distutils: language=c++
# distutils: extra_compile_args = -std=c++14
# encoding: utf-8
# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8, embedsignature=True
# cython: infer_types=True

import numpy as np

def cart_bctx(
    str task, X, y, sample_task_weight,
    xlbs, xubs, xlbs_used, xubs_used,
    feature_stds, sample_bctx_weight,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_weight_fraction_leaf,
    int max_features,
    double min_impurity_decrease,
    double min_bctx_gain,
    double bctx_lambda,
    int bctx_mode,
    unsigned int random_state
):
    cdef Constraints constraints = Constraints()
    constraints.max_depth = max_depth
    constraints.min_samples_split = min_samples_split
    constraints.min_samples_leaf = min_samples_leaf
    constraints.min_weight_leaf = min_weight_fraction_leaf * np.sum(sample_task_weight)
    constraints.max_features = max_features
    constraints.min_impurity_decrease = min_impurity_decrease
    constraints.min_bctx_gain = min_bctx_gain
    constraints.bctx_lambda = bctx_lambda
    constraints.bctx_mode = bctx_mode

    cdef ArrayRepresentedTree tree
    
    if task == 'classif':
        tree = build_tree_classif(
            X, y, sample_task_weight,
            xlbs, xubs, xlbs_used, xubs_used,
            feature_stds, sample_bctx_weight,
            constraints, random_state
        )
    else:
        tree = build_tree_regress(
            X, y, sample_task_weight,
            xlbs, xubs, xlbs_used, xubs_used,
            feature_stds, sample_bctx_weight,
            constraints, random_state
        )

    cdef dict result = {}
    result['depth'] = np.array(tree.depth, dtype=int)
    result['n_samples'] = np.array(tree.n_samples, dtype=int)
    result['impurity'] = np.array(tree.impurity, dtype=float)
    result['node_loss'] = np.array(tree.node_loss, dtype=float)
    result['subtree_loss'] = np.array(tree.subtree_loss, dtype=float)
    result['class_probs'] = np.array(
        [np.array(probs, dtype=float) for probs in tree.class_probs],
        dtype=object
    )
    result['predicted_class'] = np.array(tree.predicted_class, dtype=int)
    result['predicted_value'] = np.array(tree.predicted_value, dtype=float)
    result['splitter'] = np.array(tree.splitter, dtype=int)
    result['threshold'] = np.array(tree.threshold, dtype=float)
    result['gain'] = np.array(tree.gain, dtype=float)
    result['left_child'] = np.array(tree.left_child, dtype=int)
    result['right_child'] = np.array(tree.right_child, dtype=int)
    return result

def compute_ccp_alpha_path(tree_dict):
    cdef ArrayRepresentedTree array_tree = ArrayRepresentedTree()
    array_tree.depth = tree_dict['depth'].tolist()
    array_tree.n_samples = tree_dict['n_samples'].tolist()
    array_tree.impurity = tree_dict['impurity'].tolist()
    array_tree.node_loss = tree_dict['node_loss'].tolist()
    array_tree.subtree_loss = tree_dict['subtree_loss'].tolist()
    array_tree.class_probs = [probs.tolist() for probs in tree_dict['class_probs']]
    array_tree.predicted_class = tree_dict['predicted_class'].tolist()
    array_tree.predicted_value = tree_dict['predicted_value'].tolist()
    array_tree.splitter = tree_dict['splitter'].tolist()
    array_tree.threshold = tree_dict['threshold'].tolist()
    array_tree.gain = tree_dict['gain'].tolist()
    array_tree.left_child = tree_dict['left_child'].tolist()
    array_tree.right_child = tree_dict['right_child'].tolist()
    cdef vector[double] alphas = ccp_alpha_path(array_tree)
    return list(alphas)

def prune_tree_by_ccp_alpha(tree_dict, double ccp_alpha):
    cdef ArrayRepresentedTree array_tree = ArrayRepresentedTree()
    array_tree.depth = tree_dict['depth'].tolist()
    array_tree.n_samples = tree_dict['n_samples'].tolist()
    array_tree.impurity = tree_dict['impurity'].tolist()
    array_tree.node_loss = tree_dict['node_loss'].tolist()
    array_tree.subtree_loss = tree_dict['subtree_loss'].tolist()
    array_tree.class_probs = [probs.tolist() for probs in tree_dict['class_probs']]
    array_tree.predicted_class = tree_dict['predicted_class'].tolist()
    array_tree.predicted_value = tree_dict['predicted_value'].tolist()
    array_tree.splitter = tree_dict['splitter'].tolist()
    array_tree.threshold = tree_dict['threshold'].tolist()
    array_tree.gain = tree_dict['gain'].tolist()
    array_tree.left_child = tree_dict['left_child'].tolist()
    array_tree.right_child = tree_dict['right_child'].tolist()
    cdef ArrayRepresentedTree pruned_tree = prune_tree(array_tree, ccp_alpha)
    cdef dict result = {}
    result['depth'] = np.array(pruned_tree.depth, dtype=int)
    result['n_samples'] = np.array(pruned_tree.n_samples, dtype=int)
    result['impurity'] = np.array(pruned_tree.impurity, dtype=float)
    result['node_loss'] = np.array(pruned_tree.node_loss, dtype=float)
    result['subtree_loss'] = np.array(pruned_tree.subtree_loss, dtype=float)
    result['class_probs'] = np.array(
        [np.array(probs, dtype=float) for probs in pruned_tree.class_probs],
        dtype=object
    )
    result['predicted_class'] = np.array(pruned_tree.predicted_class, dtype=int)
    result['predicted_value'] = np.array(pruned_tree.predicted_value, dtype=float)
    result['splitter'] = np.array(pruned_tree.splitter, dtype=int)
    result['threshold'] = np.array(pruned_tree.threshold, dtype=float)
    result['gain'] = np.array(pruned_tree.gain, dtype=float)
    result['left_child'] = np.array(pruned_tree.left_child, dtype=int)
    result['right_child'] = np.array(pruned_tree.right_child, dtype=int)
    return result

def prune_tree_equal_siblings(tree_dict, double tau):
    cdef ArrayRepresentedTree array_tree = ArrayRepresentedTree()
    array_tree.depth = tree_dict['depth'].tolist()
    array_tree.n_samples = tree_dict['n_samples'].tolist()
    array_tree.impurity = tree_dict['impurity'].tolist()
    array_tree.node_loss = tree_dict['node_loss'].tolist()
    array_tree.subtree_loss = tree_dict['subtree_loss'].tolist()
    array_tree.class_probs = [probs.tolist() for probs in tree_dict['class_probs']]
    array_tree.predicted_class = tree_dict['predicted_class'].tolist()
    array_tree.predicted_value = tree_dict['predicted_value'].tolist()
    array_tree.splitter = tree_dict['splitter'].tolist()
    array_tree.threshold = tree_dict['threshold'].tolist()
    array_tree.gain = tree_dict['gain'].tolist()
    array_tree.left_child = tree_dict['left_child'].tolist()
    array_tree.right_child = tree_dict['right_child'].tolist()
    cdef ArrayRepresentedTree pruned_tree = prune_equal_siblings(array_tree, tau)
    cdef dict result = {}
    result['depth'] = np.array(pruned_tree.depth, dtype=int)
    result['n_samples'] = np.array(pruned_tree.n_samples, dtype=int)
    result['impurity'] = np.array(pruned_tree.impurity, dtype=float)
    result['node_loss'] = np.array(pruned_tree.node_loss, dtype=float)
    result['subtree_loss'] = np.array(pruned_tree.subtree_loss, dtype=float)
    result['class_probs'] = np.array(
        [np.array(probs, dtype=float) for probs in pruned_tree.class_probs],
        dtype=object
    )
    result['predicted_class'] = np.array(pruned_tree.predicted_class, dtype=int)
    result['predicted_value'] = np.array(pruned_tree.predicted_value, dtype=float)
    result['splitter'] = np.array(pruned_tree.splitter, dtype=int)
    result['threshold'] = np.array(pruned_tree.threshold, dtype=float)
    result['gain'] = np.array(pruned_tree.gain, dtype=float)
    result['left_child'] = np.array(pruned_tree.left_child, dtype=int)
    result['right_child'] = np.array(pruned_tree.right_child, dtype=int)
    return result