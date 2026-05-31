#pragma once

#include <common.hpp>

BEGIN_NAMESPACE

/*
 * Constraints for building the decision tree
 */
struct Constraints {
    int max_depth; // Maximum depth of the tree
    int min_samples_split; // Minimum samples required to split an internal node
    int min_samples_leaf; // Minimum samples required to be at a leaf node
    double min_weight_leaf; // Minimum weight required to be at a leaf node
    int max_features; // Maximum number of features to consider for splitting
    double min_impurity_decrease; // Minimum impurity decrease required to split
    double min_bctx_gain; // Minimum BCTX gain required to split
    double bctx_lambda; // Regularization parameter for BCTX
    int bctx_mode; // Mode for BCTX calculation (1:JAC, 2:JLU, 3:MAE, or 4:MSE)
    Constraints() {}
};

/*
 * Decision tree node
 */
struct Node {
    int depth; // Depth of the node in the tree
    int n_samples; // Number of samples at the node
    double impurity; // Impurity of the node
    double node_loss; // Loss at the node
    double subtree_loss; // Loss of the subtree rooted at the node
    
    Vector<double> class_probs; // Class probabilities at the node
    int predicted_class; // Predicted class at the node
    double predicted_value; // Predicted value at the node
    
    int splitter; // Feature index used for splitting
    double threshold; // Threshold value for splitting
    double gain; // Gain from the split

    std::shared_ptr<Node> left_child; // Left child node
    std::shared_ptr<Node> right_child; // Right child node

    Node() : depth(0), n_samples(0), impurity(0.0), node_loss(0.0), subtree_loss(0.0),
             class_probs(), predicted_class(-1), predicted_value(0.0),
             splitter(-1), threshold(0.0), gain(0.0),
             left_child(nullptr), right_child(nullptr) {}
};

/*
 * Tree represented as arrays for efficient storage and traversal
 */
struct ArrayRepresentedTree {
    Vector<int> depth;
    Vector<int> n_samples;
    Vector<double> impurity;
    Vector<double> node_loss;
    Vector<double> subtree_loss;
    Matrix<double> class_probs;
    Vector<int> predicted_class;
    Vector<double> predicted_value;
    Vector<int> splitter;
    Vector<double> threshold;
    Vector<double> gain;
    Vector<int> left_child;
    Vector<int> right_child;
    ArrayRepresentedTree() {}
};

/*
 * Build decision tree from ArrayRepresentedTree
 */
std::shared_ptr<Node> build_tree_from_array(const ArrayRepresentedTree& array_tree, int index = 0) {
    if (index < 0 || index >= static_cast<int>(array_tree.depth.size())) {
        return nullptr;
    }
    auto node = std::make_shared<Node>();
    node->depth = array_tree.depth[index];
    node->n_samples = array_tree.n_samples[index];
    node->impurity = array_tree.impurity[index];
    node->node_loss = array_tree.node_loss[index];
    node->subtree_loss = array_tree.subtree_loss[index];
    node->class_probs = array_tree.class_probs[index];
    node->predicted_class = array_tree.predicted_class[index];
    node->predicted_value = array_tree.predicted_value[index];
    node->splitter = array_tree.splitter[index];
    node->threshold = array_tree.threshold[index];
    node->gain = array_tree.gain[index];
    node->left_child = build_tree_from_array(array_tree, array_tree.left_child[index]);
    node->right_child = build_tree_from_array(array_tree, array_tree.right_child[index]);
    return node;
}

/*
 * Convert decision tree to ArrayRepresentedTree
 */
ArrayRepresentedTree convert_tree_to_array(const std::shared_ptr<Node>& root) {
    ArrayRepresentedTree array_tree;
    if (!root) return array_tree;

    std::function<int(const std::shared_ptr<Node>&)> count_nodes;
    count_nodes = [&](const std::shared_ptr<Node>& node) -> int {
        if (!node) return 0;
        return 1 + count_nodes(node->left_child) + count_nodes(node->right_child);
    };

    int total_nodes = count_nodes(root);
    array_tree.depth.resize(total_nodes);
    array_tree.n_samples.resize(total_nodes);
    array_tree.impurity.resize(total_nodes);
    array_tree.node_loss.resize(total_nodes);
    array_tree.subtree_loss.resize(total_nodes);
    array_tree.class_probs.resize(total_nodes);
    array_tree.predicted_class.resize(total_nodes);
    array_tree.predicted_value.resize(total_nodes);
    array_tree.splitter.resize(total_nodes, -1);
    array_tree.threshold.resize(total_nodes);
    array_tree.gain.resize(total_nodes);
    array_tree.left_child.resize(total_nodes, -1);
    array_tree.right_child.resize(total_nodes, -1);

    std::function<int(const std::shared_ptr<Node>&, int&)> fill_arrays;
    fill_arrays = [&](const std::shared_ptr<Node>& node, int& index) -> int {
        if (!node) return -1;
        int current_index = index++;
        array_tree.depth[current_index] = node->depth;
        array_tree.n_samples[current_index] = node->n_samples;
        array_tree.impurity[current_index] = node->impurity;
        array_tree.node_loss[current_index] = node->node_loss;
        array_tree.subtree_loss[current_index] = node->subtree_loss;
        array_tree.class_probs[current_index] = node->class_probs;
        array_tree.predicted_class[current_index] = node->predicted_class;
        array_tree.predicted_value[current_index] = node->predicted_value;
        array_tree.splitter[current_index] = node->splitter;
        array_tree.threshold[current_index] = node->threshold;
        array_tree.gain[current_index] = node->gain;
        array_tree.left_child[current_index] = fill_arrays(node->left_child, index);
        array_tree.right_child[current_index] = fill_arrays(node->right_child, index);
        return current_index;
    };
    
    int index = 0;
    fill_arrays(root, index);
    return array_tree;
}

END_NAMESPACE