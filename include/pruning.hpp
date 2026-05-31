#pragma once

#include <common.hpp>
#include <tree.hpp>

BEGIN_NAMESPACE

/*
 * Functions to prune decision trees
 */
std::pair<double, int> compute_complexity(
    ArrayRepresentedTree& array_tree,
    int index,
    Vector<double>& complexity
) {
    // leaf or pruned
    if (array_tree.splitter[index] == -1) {
        array_tree.subtree_loss[index] = array_tree.node_loss[index];
        return std::make_pair(array_tree.node_loss[index], 1);
    }
    // internal
    int left_index = array_tree.left_child[index];
    int right_index = array_tree.right_child[index];
    auto [left_loss, left_leaves] = compute_complexity(array_tree, left_index, complexity);
    auto [right_loss, right_leaves] = compute_complexity(array_tree, right_index, complexity);
    int n_leaves = left_leaves + right_leaves;
    double node_loss = array_tree.node_loss[index];
    array_tree.subtree_loss[index] = left_loss + right_loss;
    complexity[index] = (node_loss - left_loss - right_loss) / (n_leaves - 1);
    complexity[index] = std::max(complexity[index], 0.0);
    return std::make_pair(array_tree.subtree_loss[index], n_leaves);
}

void prune_subtree(
    ArrayRepresentedTree& array_tree,
    int index,
    Vector<double>& complexity
) {
    // leaf or pruned
    if (array_tree.splitter[index] == -1) return;
    // internal
    prune_subtree(array_tree, array_tree.left_child[index], complexity);
    prune_subtree(array_tree, array_tree.right_child[index], complexity);
    complexity[index] = std::numeric_limits<double>::infinity();
    array_tree.splitter[index] = -1;
    array_tree.left_child[index] = -1;
    array_tree.right_child[index] = -1;
}

Vector<double> ccp_alpha_path(ArrayRepresentedTree array_tree) {
    // compute cost-complexity pruning path
    int n_nodes = static_cast<int>(array_tree.depth.size());
    Vector<double> complexity(n_nodes, std::numeric_limits<double>::infinity());
    Vector<double> alphas;
    while (array_tree.splitter[0] != -1) {
        compute_complexity(array_tree, 0, complexity);
        int target_index = std::min_element(
            complexity.begin(), complexity.end()
        ) - complexity.begin();
        double alpha = complexity[target_index];
        alphas.push_back(alpha);
        prune_subtree(array_tree, target_index, complexity);
    }
    alphas.erase(
        std::unique(alphas.begin(), alphas.end()),
        alphas.end()
    );
    return alphas;
}

ArrayRepresentedTree prune_tree(
    ArrayRepresentedTree array_tree,
    double ccp_alpha
) {
    // function to recursively prune tree based on ccp_alpha
    int n_nodes = static_cast<int>(array_tree.depth.size());
    Vector<double> complexity(n_nodes, std::numeric_limits<double>::infinity());
    while (array_tree.splitter[0] != -1) {
        compute_complexity(array_tree, 0, complexity);
        int target_index = std::min_element(
            complexity.begin(), complexity.end()
        ) - complexity.begin();
        double alpha = complexity[target_index];
        if (alpha > ccp_alpha) break;
        prune_subtree(array_tree, target_index, complexity);
    }
    std::shared_ptr<Node> pruned_root = build_tree_from_array(array_tree);
    ArrayRepresentedTree pruned_array_tree = convert_tree_to_array(pruned_root);
    return pruned_array_tree;
}



ArrayRepresentedTree prune_equal_siblings(
    ArrayRepresentedTree array_tree,
    double tau = -1 // < 0 for classification tasks, >= 0 for regression tasks
) {
    // function to recursively prune siglings with equal predictions
    int n_nodes = static_cast<int>(array_tree.depth.size());
    std::function<void(int)> prune_equal_siblings;
    prune_equal_siblings = [&](int index) {
        if (array_tree.splitter[index] == -1) return;
        int left_index = array_tree.left_child[index];
        int right_index = array_tree.right_child[index];
        prune_equal_siblings(left_index);
        prune_equal_siblings(right_index);
        // prune if both children are leaves with same prediction
        if (array_tree.splitter[left_index] == -1 and
            array_tree.splitter[right_index] == -1
        ) {
            if (tau < 0) {
                // classification task
                if (array_tree.predicted_class[left_index] ==
                    array_tree.predicted_class[right_index]) {
                    array_tree.splitter[index] = -1;
                    array_tree.left_child[index] = -1;
                    array_tree.right_child[index] = -1;
                }
            } else {
                // regression task
                double left_value = array_tree.predicted_value[left_index];
                double right_value = array_tree.predicted_value[right_index];
                if (std::abs(left_value - right_value) <= tau) {
                    array_tree.splitter[index] = -1;
                    array_tree.left_child[index] = -1;
                    array_tree.right_child[index] = -1;
                }
            }
        }
    };
    prune_equal_siblings(0);
    std::shared_ptr<Node> pruned_root = build_tree_from_array(array_tree);
    ArrayRepresentedTree pruned_array_tree = convert_tree_to_array(pruned_root);
    return pruned_array_tree;
}

END_NAMESPACE