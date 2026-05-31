#pragma once

#include <common.hpp>

BEGIN_NAMESPACE

/*
 * DataHandler class to handle dataset operations
 */
template<typename t_target> class DataHandler {
    const Matrix<double>& X; // Feature matrix
    const Vector<t_target>& y;    // Target vector
    const Vector<double>& sample_task_weight; // Sample weights for loss calculation

    const Matrix<double>& xlbs; // Lower bounds for explanation
    const Matrix<double>& xubs; // Upper bounds for explanation
    const Matrix<int>& xlbs_used; // Indicator matrix for used lower bounds
    const Matrix<int>& xubs_used; // Indicator matrix for used upper bounds
    const Vector<double>& feature_stds; // Feature standard deviations for BCTX
    const Vector<double>& sample_bctx_weight; // Sample weights for BCTX regularization

    int n_samples; // Total number of samples
    int n_features; // Number of features
    int n_classes;  // Number of classes if classification

    double total_task_weight; // Total weight of all samples
    double total_bctx_weight; // Total weight for BCTX regularization
    mutable Vector<double> bctx_loss_memo; // Memoization for BCTX loss computation
    mutable Vector<int> direction_memo;

public:
    DataHandler(
        const Matrix<double>& X, // Feature matrix
        const Vector<t_target>& y, // Target vector
        const Vector<double>& sample_task_weight, // Sample weights for loss calculation
        const Matrix<double>& xlbs,
        const Matrix<double>& xubs,
        const Matrix<int>& xlbs_used,
        const Matrix<int>& xubs_used,
        const Vector<double>& feature_stds,
        const Vector<double>& sample_bctx_weight
    ) : X(X), y(y), sample_task_weight(sample_task_weight),
        xlbs(xlbs), xubs(xubs), xlbs_used(xlbs_used), xubs_used(xubs_used),
        feature_stds(feature_stds), sample_bctx_weight(sample_bctx_weight) {
        n_samples = X.size();
        n_features = X[0].size();
        n_classes = *std::max_element(y.begin(), y.end()) + 1;
        total_task_weight = std::accumulate(sample_task_weight.begin(), sample_task_weight.end(), 0.0);
        total_bctx_weight = std::accumulate(sample_bctx_weight.begin(), sample_bctx_weight.end(), 0.0);
        bctx_loss_memo.resize(n_samples, 0.0);
        direction_memo.resize(n_samples, 0);
    }

public:
    /*
     * Getters for dataset properties
     */
    double get_x(int sample_index, int feature_index) const { return X[sample_index][feature_index]; }
    t_target get_y(int index) const { return y[index]; }
    int get_n_samples() const { return n_samples; }
    int get_n_features() const { return n_features; }
    int get_n_classes() const { return n_classes; }
    double get_xlb(int sample_index, int feature_index) const { return xlbs[sample_index][feature_index]; }
    double get_xub(int sample_index, int feature_index) const { return xubs[sample_index][feature_index]; }
    int get_xlb_used(int sample_index, int feature_index) const { return xlbs_used[sample_index][feature_index]; }
    int get_xub_used(int sample_index, int feature_index) const { return xubs_used[sample_index][feature_index]; }
    double get_feature_std(int feature_index) const { return feature_stds[feature_index]; }
    double get_sample_task_weight(int index) const { return sample_task_weight[index]; }
    double get_sample_bctx_weight(int index) const { return sample_bctx_weight[index]; }
    double get_total_task_weight() const { return total_task_weight; }
    double get_total_bctx_weight() const { return total_bctx_weight; }

public:
    /*
     * Sort sample indices for each feature
     */
    Matrix<int> sort_indices_per_feature() const {
        Matrix<int> sorted_indices(n_features, Vector<int>(n_samples));
        for (int j = 0; j < n_features; ++j) {
            std::iota(sorted_indices[j].begin(), sorted_indices[j].end(), 0);
            std::sort(sorted_indices[j].begin(), sorted_indices[j].end(),
                      [&](int a, int b) { return X[a][j] < X[b][j]; });
        }
        return sorted_indices;
    }

    /*
     * Split sorted indices based on feature and threshold
     */
    std::pair<Matrix<int>, Matrix<int>> split_sorted_indices(
        const Matrix<int>& sorted_indices,
        int feature_index,
        double threshold
    ) const {
        int n_samples = sorted_indices[0].size();
        int n_left_samples = 0;
        int n_right_samples = 0;
        for (int i = 0; i < n_samples; ++i) {
            int sample_index = sorted_indices[feature_index][i];
            if (X[sample_index][feature_index] < threshold) {
                direction_memo[sample_index] = -1;
                ++n_left_samples;
            } else {
                direction_memo[sample_index] = 1;
                ++n_right_samples;
            }
        }

        Matrix<int> left_sorted_indices(n_features, Vector<int>(n_left_samples));
        Matrix<int> right_sorted_indices(n_features, Vector<int>(n_right_samples));
        for (int j = 0; j < n_features; ++j) {
            int left_pos = 0;
            int right_pos = 0;
            for (int i = 0; i < n_samples; ++i) {
                int sample_index = sorted_indices[j][i];
                if (direction_memo[sample_index] == -1) {
                    left_sorted_indices[j][left_pos] = sample_index;
                    ++left_pos;
                } else {
                    right_sorted_indices[j][right_pos] = sample_index;
                    ++right_pos;
                }
            }
        }
        return {left_sorted_indices, right_sorted_indices};
    }

public:
    /*
     * Compute initial explanation
     */
    std::pair<Vector<double>, Vector<double>> compute_initial_explanation() const {
        Vector<double> xlb_init(n_features, std::numeric_limits<double>::max());
        Vector<double> xub_init(n_features, std::numeric_limits<double>::lowest());
        for (int j = 0; j < n_features; ++j) {
            for (int i = 0; i < n_samples; ++i) {
                double val = X[i][j];
                if (val < xlb_init[j]) xlb_init[j] = val;
                if (val > xub_init[j]) xub_init[j] = val;
            }
        }
        return {xlb_init, xub_init};
    }
};

END_NAMESPACE