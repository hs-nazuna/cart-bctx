#pragma once

#include <common.hpp>
#include <tree.hpp>
#include <data_handler.hpp>
#include <gini_handler.hpp>
#include <mse_handler.hpp>
#include <bctx_handler.hpp>

BEGIN_NAMESPACE

/*
 * Structure to represent a split
 */
struct Split {
    int feature_index;   // Index of the feature to split on
    double threshold;    // Threshold value for the split
    double gain;        // Gain from the split
};

/*
 * Function to find the best split for classification tasks
 */
Split best_split_classif(
    const Vector<double>& xlb,
    const Vector<double>& xub,
    const Vector<int>& xlb_used,
    const Vector<int>& xub_used,
    const DataHandler<int>& data_handler,
    const Matrix<int>& sorted_indices,
    const Constraints& constraints,
    std::mt19937& rng
) {
    int n_samples = sorted_indices[0].size();

    // Randomly select features to consider for splitting if max_features is set
    Vector<int> feature_indices(data_handler.get_n_features());
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
    feature_indices.resize(constraints.max_features);
    
    // find the best split among the selected features
    Split best_split{-1, 0.0, 0.0};

    for (int feature_index : feature_indices) {
        double feature_std = data_handler.get_feature_std(feature_index);
        if (feature_std == 0.0) continue;

        // initialize gini handler
        GiniHandler gini_handler(data_handler.get_n_classes());
        for (int i = 0; i < n_samples; ++i) {
            int sample_index = sorted_indices[feature_index][i];
            int class_index = data_handler.get_y(sample_index);
            double clf_weight = data_handler.get_sample_task_weight(sample_index);
            gini_handler.add_to_right(class_index, clf_weight);
        }

        // initialize bctx handler
        BCTXHandler bctx_handler(constraints.bctx_mode);
        if (constraints.bctx_lambda > 0.0) {
            for (int i = 0; i < n_samples; ++i) {
                int sample_index = sorted_indices[feature_index][i];
                double bctx_weight = data_handler.get_sample_bctx_weight(sample_index);
                if (constraints.bctx_mode == JAC) {
                    // for JAC
                    int used_old = (
                        data_handler.get_xlb_used(sample_index, feature_index) |
                        data_handler.get_xub_used(sample_index, feature_index)
                    );
                    int used_now = (
                        xlb_used[feature_index] |
                        xub_used[feature_index]
                    );
                    int jac = 0;
                    if (used_old == 1 and used_now == 0) jac = 1;
                    if (used_old == 0 and used_now == 0) jac = -1;
                    bctx_handler.add_to_right(bctx_weight, jac);
                }
                else if (constraints.bctx_mode == JLU) {
                    // for JLU
                    int used_old_lb = data_handler.get_xlb_used(sample_index, feature_index);
                    int used_now_lb = xlb_used[feature_index];
                    int jac = 0;
                    if (used_old_lb == 1 and used_now_lb == 0) jac = 1;
                    if (used_old_lb == 0 and used_now_lb == 0) jac = -1;
                    bctx_handler.add_to_right(bctx_weight, jac);
                }
                else {
                    // for MAE and MSE
                    double lb = data_handler.get_xlb(sample_index, feature_index);
                    double ub = data_handler.get_xub(sample_index, feature_index);
                    bctx_handler.add_to_right(lb, bctx_weight);
                }
            }
        }

        // compute current impurity
        double current_impurity = gini_handler.total_impurity();

        for (int i = 0; i < n_samples - 1; ++i) {
            int sample_index = sorted_indices[feature_index][i];

            // update gini handler
            int class_index = data_handler.get_y(sample_index);
            double clf_weight = data_handler.get_sample_task_weight(sample_index);
            gini_handler.sub_from_right(class_index, clf_weight);
            gini_handler.add_to_left(class_index, clf_weight);
            
            // update bctx handler
            if (constraints.bctx_lambda > 0) {
                double bctx_weight = data_handler.get_sample_bctx_weight(sample_index);
                if (constraints.bctx_mode == JAC) {
                    // for JAC
                    // no update because gain is constant for JAC
                }
                else if (constraints.bctx_mode == JLU) {
                    // for JLU
                    int jac_lb = 0;
                    int used_old_lb = data_handler.get_xlb_used(sample_index, feature_index);
                    int used_now_lb = xlb_used[feature_index];
                    if (used_old_lb == 1 and used_now_lb == 0) jac_lb = 1;
                    if (used_old_lb == 0 and used_now_lb == 0) jac_lb = -1;
                    int jac_ub = 0;
                    int used_old_ub = data_handler.get_xub_used(sample_index, feature_index);
                    int used_now_ub = xub_used[feature_index];
                    if (used_old_ub == 1 and used_now_ub == 0) jac_ub = 1;
                    if (used_old_ub == 0 and used_now_ub == 0) jac_ub = -1;
                    bctx_handler.sub_from_right(bctx_weight, jac_lb);
                    bctx_handler.add_to_left(bctx_weight, jac_ub);
                }
                else {
                    // for MAE and MSE
                    double lb = data_handler.get_xlb(sample_index, feature_index);
                    double ub = data_handler.get_xub(sample_index, feature_index);
                    bctx_handler.sub_from_right(lb, bctx_weight);
                    bctx_handler.add_to_left(ub, bctx_weight);
                }
            }

            // validate split
            double x_left = data_handler.get_x(sample_index, feature_index);
            double x_right = data_handler.get_x(sorted_indices[feature_index][i + 1], feature_index);
            if (x_left == x_right) continue;

            // check leaf constraints
            int n_left_samples = i + 1;
            int n_right_samples = n_samples - n_left_samples;
            if (n_left_samples < constraints.min_samples_leaf) continue;
            if (n_right_samples < constraints.min_samples_leaf) continue;

            double left_weight = gini_handler.get_left_weight();
            double right_weight = gini_handler.get_right_weight();
            if (left_weight < constraints.min_weight_leaf) continue;
            if (right_weight < constraints.min_weight_leaf) continue;

            // compute gain
            double threshold = (x_left + x_right) / 2.0;
            double new_impurity = gini_handler.total_impurity();
            double clf_gain = current_impurity - new_impurity;
            if (clf_gain <= constraints.min_impurity_decrease) continue;

            double bctx_gain = 0.0;
            if (constraints.bctx_lambda > 0) {
                if (constraints.bctx_mode == JAC or constraints.bctx_mode == JLU) {
                    // for JAC and JLU
                    bctx_gain = bctx_handler.compute_bctx_gain() / data_handler.get_n_features();
                    if (constraints.bctx_mode == JLU) bctx_gain /= 2;
                } else {
                    // for MAE and MSE
                    bctx_gain = (
                        bctx_handler.compute_bctx_measure(xub[feature_index], xlb[feature_index]) -
                        bctx_handler.compute_bctx_measure(threshold, threshold)
                    ) / (data_handler.get_n_features() * 2);
                    if (constraints.bctx_mode == MAE) bctx_gain /= feature_std;
                    if (constraints.bctx_mode == MSE) bctx_gain /= square(feature_std);
                }
                if (bctx_gain <= constraints.min_bctx_gain) continue;
            }

            // update best split
            double gain = clf_gain + constraints.bctx_lambda * bctx_gain;
            if (gain > best_split.gain) best_split = Split{feature_index, threshold, gain};
        }
    }
    
    return best_split;
}

/*
 * Function to find the best split for regression tasks
 */
Split best_split_regress(
    const Vector<double>& xlb,
    const Vector<double>& xub,
    const Vector<int>& xlb_used,
    const Vector<int>& xub_used,
    const DataHandler<double>& data_handler,
    const Matrix<int>& sorted_indices,
    const Constraints& constraints,
    std::mt19937& rng
) {
    int n_samples = sorted_indices[0].size();

    // Randomly select features to consider for splitting if max_features is set
    Vector<int> feature_indices(data_handler.get_n_features());
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
    feature_indices.resize(constraints.max_features);
    
    // find the best split among the selected features
    Split best_split{-1, 0.0, 0.0};

    for (int feature_index : feature_indices) {
        double feature_std = data_handler.get_feature_std(feature_index);
        if (feature_std == 0.0) continue;

        // initialize MSE manager
        MSEHandler mse_handler;
        for (int i = 0; i < n_samples; ++i) {
            int sample_index = sorted_indices[feature_index][i];
            double sample_value = data_handler.get_y(sample_index);
            double reg_weight = data_handler.get_sample_task_weight(sample_index);
            mse_handler.add_to_right(sample_value, reg_weight);
        }

        // initialize bctx handler
        BCTXHandler bctx_handler(constraints.bctx_mode);
        if (constraints.bctx_lambda > 0.0) {
            for (int i = 0; i < n_samples; ++i) {
                int sample_index = sorted_indices[feature_index][i];
                double bctx_weight = data_handler.get_sample_bctx_weight(sample_index);
                if (constraints.bctx_mode == JAC) {
                    // for JAC
                    int used_old = (
                        data_handler.get_xlb_used(sample_index, feature_index) |
                        data_handler.get_xub_used(sample_index, feature_index)
                    );
                    int used_now = (
                        xlb_used[feature_index] |
                        xub_used[feature_index]
                    );
                    int jac = 0;
                    if (used_old == 1 and used_now == 0) jac = 1;
                    if (used_old == 0 and used_now == 0) jac = -1;
                    bctx_handler.add_to_right(bctx_weight, jac);
                }
                else if (constraints.bctx_mode == JLU) {
                    // for JLU
                    int used_old_lb = data_handler.get_xlb_used(sample_index, feature_index);
                    int used_now_lb = xlb_used[feature_index];
                    int jac = 0;
                    if (used_old_lb == 1 and used_now_lb == 0) jac = 1;
                    if (used_old_lb == 0 and used_now_lb == 0) jac = -1;
                    bctx_handler.add_to_right(bctx_weight, jac);
                }
                else {
                    // for MAE and MSE
                    double lb = data_handler.get_xlb(sample_index, feature_index);
                    double ub = data_handler.get_xub(sample_index, feature_index);
                    bctx_handler.add_to_right(lb, bctx_weight);
                }
            }
        }

        // compute current impurity
        double mean_value = mse_handler.get_mean_value();
        double current_impurity = mse_handler.compute_total_mse(mean_value, mean_value);
        
        for (int i = 0; i < n_samples - 1; ++i) {
            int sample_index = sorted_indices[feature_index][i];

            // update mse handler
            double sample_value = data_handler.get_y(sample_index);
            double reg_weight = data_handler.get_sample_task_weight(sample_index);
            mse_handler.sub_from_right(sample_value, reg_weight);
            mse_handler.add_to_left(sample_value, reg_weight);

            // update bctx handler
            if (constraints.bctx_lambda > 0) {
                double bctx_weight = data_handler.get_sample_bctx_weight(sample_index);
                if (constraints.bctx_mode == JAC) {
                    // for JAC
                    // no update because gain is constant for JAC
                }
                else if (constraints.bctx_mode == JLU) {
                    // for JLU
                    int jac_lb = 0;
                    int used_old_lb = data_handler.get_xlb_used(sample_index, feature_index);
                    int used_now_lb = xlb_used[feature_index];
                    if (used_old_lb == 1 and used_now_lb == 0) jac_lb = 1;
                    if (used_old_lb == 0 and used_now_lb == 0) jac_lb = -1;
                    int jac_ub = 0;
                    int used_old_ub = data_handler.get_xub_used(sample_index, feature_index);
                    int used_now_ub = xub_used[feature_index];
                    if (used_old_ub == 1 and used_now_ub == 0) jac_ub = 1;
                    if (used_old_ub == 0 and used_now_ub == 0) jac_ub = -1;
                    bctx_handler.sub_from_right(bctx_weight, jac_lb);
                    bctx_handler.add_to_left(bctx_weight, jac_ub);
                }
                else {
                    // for MAE and MSE
                    double lb = data_handler.get_xlb(sample_index, feature_index);
                    double ub = data_handler.get_xub(sample_index, feature_index);
                    bctx_handler.sub_from_right(lb, bctx_weight);
                    bctx_handler.add_to_left(ub, bctx_weight);
                }
            }

            // check if this is a valid split
            double x_left = data_handler.get_x(sample_index, feature_index);
            double x_right = data_handler.get_x(sorted_indices[feature_index][i + 1], feature_index);
            if (x_left == x_right) continue;

            // check leaf constraints
            int n_left_samples = i + 1;
            int n_right_samples = n_samples - n_left_samples;
            if (n_left_samples < constraints.min_samples_leaf) continue;
            if (n_right_samples < constraints.min_samples_leaf) continue;

            double left_weight = mse_handler.get_left_weight();
            double right_weight = mse_handler.get_right_weight();
            double total_weight = left_weight + right_weight;
            if (left_weight < constraints.min_weight_leaf) continue;
            if (right_weight < constraints.min_weight_leaf) continue;

            // compute gain
            double threshold = (x_left + x_right) / 2.0;
            double left_mean = mse_handler.get_left_mean();
            double right_mean = mse_handler.get_right_mean();
            double new_impurity = mse_handler.compute_total_mse(left_mean, right_mean);
            double reg_gain = current_impurity - new_impurity;
            if (reg_gain <= constraints.min_impurity_decrease) continue;

            double bctx_gain = 0.0;
            if (constraints.bctx_lambda > 0) {
                if (constraints.bctx_mode == JAC or constraints.bctx_mode == JLU) {
                    // for JAC and JLU
                    bctx_gain = bctx_handler.compute_bctx_gain() / data_handler.get_n_features();
                    if (constraints.bctx_mode == JLU) bctx_gain /= 2;
                } else {
                    // for MAE and MSE
                    bctx_gain = (
                        bctx_handler.compute_bctx_measure(xub[feature_index], xlb[feature_index]) -
                        bctx_handler.compute_bctx_measure(threshold, threshold)
                    ) / (data_handler.get_n_features() * 2);
                    if (constraints.bctx_mode == MAE) bctx_gain /= feature_std;
                    if (constraints.bctx_mode == MSE) bctx_gain /= square(feature_std);
                }
                if (bctx_gain <= constraints.min_bctx_gain) continue;
            }

            // update best split
            double gain = reg_gain + constraints.bctx_lambda * bctx_gain;
            if (gain > best_split.gain) best_split = Split{feature_index, threshold, gain};
        }
    }

    return best_split;
}

END_NAMESPACE