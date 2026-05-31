#pragma once

#include <common.hpp>

BEGIN_NAMESPACE

/*
 * MSEHandler class to handle MSE calculations for regression tasks
 */
class MSEHandler {
    double left_weight;  // Total weight for the left node
    double right_weight; // Total weight for the right node
    double left_sum;    // Sum of target values for the left node
    double right_sum;   // Sum of target values for the right node
    double left_sq_sum;  // Sum of squares of target values for the left node
    double right_sq_sum; // Sum of squares of target values for the right node

public:
    MSEHandler()
        : left_weight(0.0), right_weight(0.0),
          left_sum(0.0), right_sum(0.0),
          left_sq_sum(0.0), right_sq_sum(0.0) {}

public:
    /*
     * Getters for means and weights
     */
    double get_left_mean() const {
        if (left_weight == 0.0) return 0.0;
        return left_sum / left_weight;
    }
    
    double get_right_mean() const {
        if (right_weight == 0.0) return 0.0;
        return right_sum / right_weight;
    }

    double get_mean_value() const {
        double total_weight = left_weight + right_weight;
        if (total_weight == 0.0) return 0.0;
        return (left_sum + right_sum) / total_weight;
    }

    double get_left_weight() const { return left_weight; }
    double get_right_weight() const { return right_weight; }

public:
    /*
     * Methods to update sums and weights for left and right samples
     */
    void add_to_left(double target_value, double weight) {
        left_weight += weight;
        left_sum += target_value * weight;
        left_sq_sum += square(target_value) * weight;
    }

    void add_to_right(double target_value, double weight) {
        right_weight += weight;
        right_sum += target_value * weight;
        right_sq_sum += square(target_value) * weight;
    }

    void sub_from_left(double target_value, double weight) {
        left_weight -= weight;
        left_sum -= target_value * weight;
        left_sq_sum -= square(target_value) * weight;
    }

    void sub_from_right(double target_value, double weight) {
        right_weight -= weight;
        right_sum -= target_value * weight;
        right_sq_sum -= square(target_value) * weight;
    }

public:
    /*
     * Methods to compute MSE for left, right, and total samples
     */
    double compute_left_mse(double value) const {
        if (left_weight == 0.0) return 0.0;
        return (left_sq_sum - 2 * value * left_sum + square(value) * left_weight) / left_weight;
    }

    double compute_right_mse(double value) const {
        if (right_weight == 0.0) return 0.0;
        return (right_sq_sum - 2 * value * right_sum + square(value) * right_weight) / right_weight;
    }

    double compute_total_mse(double left_value, double right_value) const {
        if (left_weight + right_weight == 0.0) return 0.0;
        double left_mse = compute_left_mse(left_value);
        double right_mse = compute_right_mse(right_value);
        return (left_mse * left_weight + right_mse * right_weight) / (left_weight + right_weight);
    }
};

//

END_NAMESPACE