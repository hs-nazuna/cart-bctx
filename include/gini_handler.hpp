#pragma once

#include <common.hpp>

BEGIN_NAMESPACE

/*
 * GiniHandler class to manage Gini impurity calculations
 */
class GiniHandler {
    double left_weight;  // Total weight for the left node
    double right_weight; // Total weight for the right node
    double sq_sum_left;  // Sum of squares of class counts for the left node
    double sq_sum_right; // Sum of squares of class counts for the right node
    Vector<double> left_counts;  // Class counts for the left node
    Vector<double> right_counts; // Class counts for the right node

public:
    GiniHandler(int n_classes)
        : left_counts(n_classes, 0.0), right_counts(n_classes, 0.0),
          left_weight(0.0), right_weight(0.0),
          sq_sum_left(0.0), sq_sum_right(0.0) {}

public:
    /*
     * Getters for weights and class counts
     */
    double get_left_weight() const { return left_weight; }
    double get_right_weight() const { return right_weight; }
    const Vector<double>& get_left_counts() const { return left_counts; }
    const Vector<double>& get_right_counts() const { return right_counts; }

public:
    /*
     * Methods to update counts and weights for left and right samples
     */
    void add_to_left(int class_index, double weight) {
        left_weight += weight;
        sq_sum_left += square(weight) + 2 * weight * left_counts[class_index];
        left_counts[class_index] += weight;
    }

    void add_to_right(int class_index, double weight) {
        right_weight += weight;
        sq_sum_right += square(weight) + 2 * weight * right_counts[class_index];
        right_counts[class_index] += weight;
    }

    void sub_from_left(int class_index, double weight) {
        left_weight -= weight;
        sq_sum_left += square(weight) - 2 * weight * left_counts[class_index];
        left_counts[class_index] -= weight;
    }

    void sub_from_right(int class_index, double weight) {
        right_weight -= weight;
        sq_sum_right += square(weight) - 2 * weight * right_counts[class_index];
        right_counts[class_index] -= weight;
    }

public:
    /*
     * Methods to compute impurities for left, right, and total samples
     */
    double left_impurity() const {
        if (left_weight == 0.0) return 0.0;
        return 1.0 - sq_sum_left / square(left_weight);
    }

    double right_impurity() const {
        if (right_weight == 0.0) return 0.0;
        return 1.0 - sq_sum_right / square(right_weight);
    }

    double total_impurity() const {
        double total_weight = left_weight + right_weight;
        if (total_weight == 0.0) return 0.0;
        return (left_weight * left_impurity() + right_weight * right_impurity()) / total_weight;
    }
};

END_NAMESPACE