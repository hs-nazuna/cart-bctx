#pragma once

#include <common.hpp>
#include <tree.hpp>
#include <data_handler.hpp>
#include <split.hpp>

BEGIN_NAMESPACE

/*
 * Function to compute bctx loss for a given sample set
 */
template<typename T> double partial_bctx_loss(
    int bctx_mode,
    const Vector<double>& xlb,
    const Vector<double>& xub,
    const Vector<int>& xlb_used,
    const Vector<int>& xub_used,
    const DataHandler<T>& data_handler,
    const Matrix<int>& sorted_indices
) {
    if (data_handler.get_total_bctx_weight() == 0.0) return 0.0;
    double bctx_loss = 0.0;
    int n_samples = sorted_indices[0].size();
    int n_features = data_handler.get_n_features();
    for (int i = 0; i < n_samples; ++i) {
        int sample_index = sorted_indices[0][i];
        double sample_bctx_weight = data_handler.get_sample_bctx_weight(sample_index);
        if (sample_bctx_weight == 0.0) continue;
        double dissimilarity = 0.0;
        for (int j = 0; j < n_features; ++j) {
            if (bctx_mode == JAC) {
                int used_old = (
                    data_handler.get_xlb_used(sample_index, j) |
                    data_handler.get_xub_used(sample_index, j)
                );
                int used_now = (xlb_used[j] | xub_used[j]);
                dissimilarity += abs(used_old - used_now);
            }
            else if (bctx_mode == JLU) {
                int used_old_lb = data_handler.get_xlb_used(sample_index, j);
                dissimilarity += abs(used_old_lb - xlb_used[j]) / 2;
                int used_old_ub = data_handler.get_xub_used(sample_index, j);
                dissimilarity += abs(used_old_ub - xub_used[j]) / 2;
            }
            else if (bctx_mode == MAE) {
                double feature_std = data_handler.get_feature_std(j);
                if (feature_std == 0.0) continue;
                double xlb_ij = data_handler.get_xlb(sample_index, j);
                double xub_ij = data_handler.get_xub(sample_index, j);
                dissimilarity += (
                    abs(xub_ij - xub[j]) / feature_std +
                    abs(xlb_ij - xlb[j]) / feature_std
                ) / 2;
            }
            else {
                double feature_std = data_handler.get_feature_std(j);
                if (feature_std == 0.0) continue;
                double xlb_ij = data_handler.get_xlb(sample_index, j);
                double xub_ij = data_handler.get_xub(sample_index, j);
                dissimilarity += (
                    square((xub_ij - xub[j]) / feature_std) +
                    square((xlb_ij - xlb[j]) / feature_std)
                ) / 2;
            }
        }
        bctx_loss += sample_bctx_weight * dissimilarity / n_features;
    }
    return bctx_loss / data_handler.get_total_bctx_weight();
}

/*
 * Functions to build classification tree
 */
std::shared_ptr<Node> build_tree_classif_recur(
    int depth,
    Vector<double>& xlb,
    Vector<double>& xub,
    Vector<int>& xlb_used,
    Vector<int>& xub_used,
    const DataHandler<int>& data_handler,
    const Matrix<int>& sorted_indices,
    const Constraints& constraints,
    std::mt19937& rng
) {    
    // compute current class probabilities
    int n_samples = sorted_indices[0].size();
    Vector<double> class_probs(data_handler.get_n_classes(), 0.0);
    double total_weight = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int sample_index = sorted_indices[0][i];
        int class_index = data_handler.get_y(sample_index);
        double weight = data_handler.get_sample_task_weight(sample_index);
        class_probs[class_index] += weight;
        total_weight += weight;
    }
    for (double& count : class_probs) {
        count /= total_weight;
    }
    
    // compute impurity
    double impurity = 0.0;
    if (total_weight > 0.0) {
        impurity = 1.0;
        for (double prob : class_probs) {
            impurity -= prob * prob;
        }
    }

    // compute classification loss
    int predicted_class = argmax(class_probs);
    double clf_loss = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int sample_index = sorted_indices[0][i];
        int class_index = data_handler.get_y(sample_index);
        double weight = data_handler.get_sample_task_weight(sample_index);
        if (class_index != predicted_class) clf_loss += weight;
    }
    clf_loss /= data_handler.get_total_task_weight();

    // compute bctx loss
    double bctx_loss = 0.0;
    if (constraints.bctx_lambda > 0.0) {
        bctx_loss = constraints.bctx_lambda * partial_bctx_loss(
            constraints.bctx_mode,
            xlb, xub, xlb_used, xub_used,
            data_handler, sorted_indices
        );
    }

    // create node
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->depth = depth;
    node->n_samples = n_samples;
    node->impurity = impurity;
    node->node_loss = clf_loss + bctx_loss;
    node->subtree_loss = node->node_loss;
    node->class_probs = class_probs;
    node->predicted_class = predicted_class;

    // check stopping criteria
    if (depth >= constraints.max_depth ||
        n_samples < constraints.min_samples_split ||
        impurity == 0.0) {
        return node;
    }

    // try the best split
    Split best_split = best_split_classif(
        xlb, xub, xlb_used, xub_used,
        data_handler, sorted_indices, constraints, rng
    );
    if (best_split.feature_index == -1) {
        return node;
    }

    auto [left_sorted_indices, right_sorted_indices] = data_handler.split_sorted_indices(
        sorted_indices, best_split.feature_index, best_split.threshold
    );

    double tmp = xub[best_split.feature_index];
    int tmp_used = xub_used[best_split.feature_index];
    xub[best_split.feature_index] = best_split.threshold;
    xub_used[best_split.feature_index] = 1;
    std::shared_ptr<Node> left_child = build_tree_classif_recur(
        depth + 1, xlb, xub, xlb_used, xub_used,
        data_handler, left_sorted_indices, constraints, rng
    );
    xub[best_split.feature_index] = tmp;
    xub_used[best_split.feature_index] = tmp_used;

    tmp = xlb[best_split.feature_index];
    tmp_used = xlb_used[best_split.feature_index];
    xlb[best_split.feature_index] = best_split.threshold;
    xlb_used[best_split.feature_index] = 1;
    std::shared_ptr<Node> right_child = build_tree_classif_recur(
        depth + 1, xlb, xub, xlb_used, xub_used,
        data_handler, right_sorted_indices, constraints, rng
    );
    xlb[best_split.feature_index] = tmp;
    xlb_used[best_split.feature_index] = tmp_used;

    // update node with children
    node->splitter = best_split.feature_index;
    node->threshold = best_split.threshold;
    node->gain = best_split.gain;
    node->subtree_loss = left_child->subtree_loss + right_child->subtree_loss;
    node->left_child = left_child;
    node->right_child = right_child;

    return node;
}

ArrayRepresentedTree build_tree_classif(
    const Matrix<double>& X,
    const Vector<int>& y,
    const Vector<double>& sample_clf_weight,
    const Matrix<double>& xlbs,
    const Matrix<double>& xubs,
    const Matrix<int>& xlbs_used,
    const Matrix<int>& xubs_used,
    const Vector<double>& feature_stds,
    const Vector<double>& sample_bctx_weight,
    const Constraints& constraints,
    unsigned int random_state
) {
    DataHandler<int> data_handler(
        X, y, sample_clf_weight,
        xlbs, xubs, xlbs_used, xubs_used,
        feature_stds, sample_bctx_weight
    );
    Matrix<int> sorted_indices = data_handler.sort_indices_per_feature();
    auto [xlb_init, xub_init] = data_handler.compute_initial_explanation();
    Vector<int> xlb_used_init(data_handler.get_n_features(), 0);
    Vector<int> xub_used_init(data_handler.get_n_features(), 0);
    std::mt19937 rng(random_state);
    std::shared_ptr<Node> root = build_tree_classif_recur(
        0, xlb_init, xub_init, xlb_used_init, xub_used_init,
        data_handler, sorted_indices, constraints, rng
    );
    return convert_tree_to_array(root);
}

/*
 * Functions to build regression tree
 */
std::shared_ptr<Node> build_tree_regress_recur(
    int depth,
    Vector<double>& xlb,
    Vector<double>& xub,
    Vector<int>& xlb_used,
    Vector<int>& xub_used,
    const DataHandler<double>& data_handler,
    const Matrix<int>& sorted_indices,
    const Constraints& constraints,
    std::mt19937& rng
) {
    // compute current mean value
    int n_samples = sorted_indices[0].size();
    double total_weight = 0.0;
    double weighted_sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int sample_index = sorted_indices[0][i];
        double target_value = data_handler.get_y(sample_index);
        double weight = data_handler.get_sample_task_weight(sample_index);
        weighted_sum += target_value * weight;
        total_weight += weight;
    }
    double mean_value = (total_weight > 0.0) ? (weighted_sum / total_weight) : 0.0;

    // compute regression loss
    double reg_loss = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int sample_index = sorted_indices[0][i];
        double target_value = data_handler.get_y(sample_index);
        double weight = data_handler.get_sample_task_weight(sample_index);
        reg_loss += weight * square(target_value - mean_value);
    }
    reg_loss /= data_handler.get_total_task_weight();

    // compute bctx loss
    double bctx_loss = 0.0;
    if (constraints.bctx_lambda > 0.0) {
        bctx_loss = constraints.bctx_lambda * partial_bctx_loss(
            constraints.bctx_mode,
            xlb, xub, xlb_used, xub_used,
            data_handler, sorted_indices
        );
    }

    // create node
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->depth = depth;
    node->n_samples = n_samples;
    node->impurity = reg_loss;
    node->node_loss = reg_loss + bctx_loss;
    node->subtree_loss = node->node_loss;
    node->predicted_value = mean_value;

    // check stopping criteria
    if (depth >= constraints.max_depth ||
        n_samples < constraints.min_samples_split ||
        reg_loss == 0.0) {
        return node;
    }

    // try the best split
    Split best_split = best_split_regress(
        xlb, xub, xlb_used, xub_used,
        data_handler, sorted_indices, constraints, rng
    );
    if (best_split.feature_index == -1) {
        return node;
    }

    auto [left_sorted_indices, right_sorted_indices] = data_handler.split_sorted_indices(
        sorted_indices, best_split.feature_index, best_split.threshold
    );

    double tmp = xub[best_split.feature_index];
    int tmp_used = xub_used[best_split.feature_index];
    xub[best_split.feature_index] = best_split.threshold;
    xub_used[best_split.feature_index] = 1;
    std::shared_ptr<Node> left_child = build_tree_regress_recur(
        depth + 1, xlb, xub, xlb_used, xub_used,
        data_handler, left_sorted_indices, constraints, rng
    );
    xub[best_split.feature_index] = tmp;
    xub_used[best_split.feature_index] = tmp_used;

    tmp = xlb[best_split.feature_index];
    tmp_used = xlb_used[best_split.feature_index];
    xlb[best_split.feature_index] = best_split.threshold;
    xlb_used[best_split.feature_index] = 1;
    std::shared_ptr<Node> right_child = build_tree_regress_recur(
        depth + 1, xlb, xub, xlb_used, xub_used,
        data_handler, right_sorted_indices, constraints, rng
    );
    xlb[best_split.feature_index] = tmp;
    xlb_used[best_split.feature_index] = tmp_used;

    // update node with children
    node->splitter = best_split.feature_index;
    node->threshold = best_split.threshold;
    node->gain = best_split.gain;
    node->subtree_loss = left_child->subtree_loss + right_child->subtree_loss;
    node->left_child = left_child;
    node->right_child = right_child;

    return node;
}

ArrayRepresentedTree build_tree_regress(
    const Matrix<double>& X,
    const Vector<double>& y,
    const Vector<double>& sample_reg_weight,
    const Matrix<double>& xlbs,
    const Matrix<double>& xubs,
    const Matrix<int>& xlbs_used,
    const Matrix<int>& xubs_used,
    const Vector<double>& feature_stds,
    const Vector<double>& sample_bctx_weight,
    const Constraints& constraints,
    unsigned int random_state
) {
    DataHandler<double> data_handler(
        X, y, sample_reg_weight,
        xlbs, xubs, xlbs_used, xubs_used,
        feature_stds, sample_bctx_weight
    );
    Matrix<int> sorted_indices = data_handler.sort_indices_per_feature();
    auto [xlb_init, xub_init] = data_handler.compute_initial_explanation();
    Vector<int> xlb_used_init(data_handler.get_n_features(), 0);
    Vector<int> xub_used_init(data_handler.get_n_features(), 0);
    std::mt19937 rng(random_state);
    std::shared_ptr<Node> root = build_tree_regress_recur(
        0, xlb_init, xub_init, xlb_used_init, xub_used_init,
        data_handler, sorted_indices, constraints, rng
    );
    return convert_tree_to_array(root);
}

END_NAMESPACE