#pragma once

#include <common.hpp>
#include <treap.hpp>

BEGIN_NAMESPACE

// BCTX mode constants
const int JAC = 1;
const int JLU = 2;
const int MAE = 3;
const int MSE = 4;

/*
 * BCTXHandler class to handle BCTX measure calculations
 */
class BCTXHandler {
    // mode indicator
    int bctx_mode;

    // common for all modes
    double left_weight;  // Total weight for the left node
    double right_weight; // Total weight for the right node

    // for JAC and JLU
    double left_jac_weight;
    double right_jac_weight;
    
    // for MAE
    Treap left_treap;   // Treap for left node
    Treap right_treap;  // Treap for right node
    
    // for MSE
    double left_sum;    // Sum of target values for the left node
    double right_sum;   // Sum of target values for the right node
    double left_sq_sum;  // Sum of squares of target values for the left node
    double right_sq_sum; // Sum of squares of target values for the right node

public:
    BCTXHandler(int bctx_mode=MSE)
        : bctx_mode(bctx_mode),
          left_weight(0.0), right_weight(0.0),
          left_jac_weight(0.0), right_jac_weight(0.0),
          left_sum(0.0), right_sum(0.0),
          left_sq_sum(0.0), right_sq_sum(0.0) {}

public:
    /*
     * Weight update methods for JAC and JLU
     */
    void add_to_left(double weight, int jac) {
        left_weight += weight;
        left_jac_weight += jac * weight;
    }

    void add_to_right(double weight, int jac) {
        right_weight += weight;
        right_jac_weight += jac * weight;
    }

    void sub_from_left(double weight, int jac) {
        left_weight -= weight;
        left_jac_weight -= jac * weight;
    }

    void sub_from_right(double weight, int jac) {
        right_weight -= weight;
        right_jac_weight -= jac * weight;
    }

    /*
     * Compute BCTX gain for JAC and JLU
     */
    double compute_bctx_gain() {
        return (left_jac_weight + right_jac_weight) / (left_weight + right_weight);
    }

public:
    /*
     * Methods to update sums and weights for left and right samples for MAE and MSE
     */
    void add_to_left(double value, double weight) {
        left_weight += weight;
        left_sum += value * weight;
        if (bctx_mode == MAE) left_treap.insert(value, weight);
        else left_sq_sum += square(value) * weight;
    }

    void add_to_right(double value, double weight) {
        right_weight += weight;
        right_sum += value * weight;
        if (bctx_mode == MAE) right_treap.insert(value, weight);
        else right_sq_sum += square(value) * weight;
    }

    void sub_from_left(double value, double weight) {
        left_weight -= weight;
        left_sum -= value * weight;
        if (bctx_mode == MAE) left_treap.erase(value, weight);
        else left_sq_sum -= square(value) * weight;
    }

    void sub_from_right(double value, double weight) {
        right_weight -= weight;
        right_sum -= value * weight;
        if (bctx_mode == MAE) right_treap.erase(value, weight);
        else right_sq_sum -= square(value) * weight;
    }

    /*
     * Methods to compute MAE for left, right, and total samples
     */
    double compute_left_mae(double value) {
        if (left_weight == 0.0) return 0.0;
        double leq_value = left_treap.prefix_value_sum(value);
        double leq_weight = left_treap.prefix_weight_sum(value);
        double ge_value = left_sum - leq_value;
        double ge_weight = left_weight - leq_weight;
        double leq_mae = (value * leq_weight - leq_value);
        double ge_mae = (ge_value - value * ge_weight);
        return (leq_mae + ge_mae) / left_weight;
    }

    double compute_right_mae(double value) {
        if (right_weight == 0.0) return 0.0;
        double leq_value = right_treap.prefix_value_sum(value);
        double leq_weight = right_treap.prefix_weight_sum(value);
        double ge_value = right_sum - leq_value;
        double ge_weight = right_weight - leq_weight;
        double leq_mae = (value * leq_weight - leq_value);
        double ge_mae = (ge_value - value * ge_weight);
        return (leq_mae + ge_mae) / right_weight;
    }

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

    /*
     * Compute total BCTX measure (MAE or MSE)
     */
    double compute_bctx_measure(double left_value, double right_value) {
        if (left_weight + right_weight == 0.0) return 0.0;
        if (bctx_mode == MAE) {
            double left_mae = compute_left_mae(left_value);
            double right_mae = compute_right_mae(right_value);
            return (left_mae * left_weight + right_mae * right_weight) / (left_weight + right_weight);
        } else {
            double left_mse = compute_left_mse(left_value);
            double right_mse = compute_right_mse(right_value);
            return (left_mse * left_weight + right_mse * right_weight) / (left_weight + right_weight);
        }
    }
};

//

END_NAMESPACE