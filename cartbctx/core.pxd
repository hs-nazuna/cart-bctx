# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8, embedsignature=True
# cython: infer_types=True

from libcpp.vector cimport vector

cdef extern from "tree.hpp" namespace "CART_BCTX_NAMESPACE":
    cdef cppclass Constraints:
        Constraints() except +
        int max_depth
        int min_samples_split
        int min_samples_leaf
        double min_weight_leaf
        int max_features
        double min_impurity_decrease
        double min_bctx_gain
        double bctx_lambda
        int bctx_mode # 1:JAC, 2:JLU, 3:MAE, 4:MSE

    cdef cppclass ArrayRepresentedTree:
        ArrayRepresentedTree() except +
        vector[int] depth
        vector[int] n_samples
        vector[double] impurity
        vector[double] node_loss
        vector[double] subtree_loss
        vector[vector[double]] class_probs
        vector[int] predicted_class
        vector[double] predicted_value
        vector[int] splitter
        vector[double] threshold
        vector[double] gain
        vector[int] left_child
        vector[int] right_child

cdef extern from "tree_builder.hpp" namespace "CART_BCTX_NAMESPACE":
    cdef ArrayRepresentedTree build_tree_classif(
        const vector[vector[double]]& X,
        const vector[int]& y,
        const vector[double]& sample_clf_weight,
        const vector[vector[double]]& xlbs,
        const vector[vector[double]]& xubs,
        const vector[vector[int]]& xlbs_used,
        const vector[vector[int]]& xubs_used,
        const vector[double]& feature_stds,
        const vector[double]& sample_bctx_weight,
        const Constraints& constraints,
        unsigned int random_state
    ) except +

    cdef ArrayRepresentedTree build_tree_regress(
        const vector[vector[double]]& X,
        const vector[double]& y,
        const vector[double]& sample_reg_weight,
        const vector[vector[double]]& xlbs,
        const vector[vector[double]]& xubs,
        const vector[vector[int]]& xlbs_used,
        const vector[vector[int]]& xubs_used,
        const vector[double]& feature_stds,
        const vector[double]& sample_bctx_weight,
        const Constraints& constraints,
        unsigned int random_state
    ) except +

cdef extern from "pruning.hpp" namespace "CART_BCTX_NAMESPACE":
    cdef vector[double] ccp_alpha_path(ArrayRepresentedTree array_tree) except +

    cdef ArrayRepresentedTree prune_tree(
        ArrayRepresentedTree array_tree,
        double ccp_alpha
    ) except +

    cdef ArrayRepresentedTree prune_equal_siblings(
        ArrayRepresentedTree array_tree,
        double tau
    ) except +