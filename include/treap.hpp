#pragma once

#include <common.hpp>

BEGIN_NAMESPACE

/*
 * Treap data structure to support insertion, deletion, and prefix sum queries
 */
class Treap {
private:
    struct TreapNode {
        double key;
        long long priority;
        int count;
        double node_weight;
        double subtree_value;
        double subtree_weight;
        std::unique_ptr<TreapNode> l, r;

        TreapNode(double k, double weight) : key(k), l(nullptr), r(nullptr) {
            static std::mt19937 mt(std::random_device{}());
            priority = mt();
            count = 1;
            node_weight = weight;
            subtree_value = k * weight;
            subtree_weight = weight;
        }
    };

    std::unique_ptr<TreapNode> root;

    // get subtree value
    double get_subtree_value(const std::unique_ptr<TreapNode>& t) {
        return t ? t->subtree_value : 0;
    }

    // get subtree weight
    double get_subtree_weight(const std::unique_ptr<TreapNode>& t) {
        return t ? t->subtree_weight : 0.0;
    }

    // update subtree sum
    void update(std::unique_ptr<TreapNode>& t) {
        if (t) {
            t->subtree_value = get_subtree_value(t->l) + get_subtree_value(t->r) + (t->key * t->node_weight);
            t->subtree_weight = get_subtree_weight(t->l) + get_subtree_weight(t->r) + t->node_weight;
        }
    }

    // split treap into two treaps: keys <= k and keys > k
    void split(
        std::unique_ptr<TreapNode> t, double k,
        std::unique_ptr<TreapNode>& l, std::unique_ptr<TreapNode>& r
    ) {
        if (!t) {
            l = nullptr;
            r = nullptr;
            return;
        }
        if (t->key <= k) {
            split(std::move(t->r), k, t->r, r);
            l = std::move(t);
            update(l);
        } else {
            split(std::move(t->l), k, l, t->l);
            r = std::move(t);
            update(r);
        }
    }

    // split treap into three treaps: keys < k, keys == k, keys > k
    void split(
        std::unique_ptr<TreapNode> t, double k,
        std::unique_ptr<TreapNode>& l, std::unique_ptr<TreapNode>& m, std::unique_ptr<TreapNode>& r
    ) {
        if (!t) {
            l = nullptr;
            m = nullptr;
            r = nullptr;
            return;
        }
        if (t->key < k) {
            split(std::move(t->r), k, t->r, m, r);
            l = std::move(t);
            update(l);
        } else if (t->key == k) {
            l = std::move(t->l);
            r = std::move(t->r);
            m = std::move(t);
            m->l = nullptr;
            m->r = nullptr;
            update(m);
        } else {
            split(std::move(t->l), k, l, m, t->l);
            r = std::move(t);
            update(r);
        }
    }

    // merge two treaps
    std::unique_ptr<TreapNode> merge(std::unique_ptr<TreapNode> l, std::unique_ptr<TreapNode> r) {
        if (!l || !r) return l ? std::move(l) : std::move(r);
        if (l->priority > r->priority) {
            l->r = merge(std::move(l->r), std::move(r));
            update(l);
            return l;
        } else {
            r->l = merge(std::move(l), std::move(r->l));
            update(r);
            return r;
        }
    }

    // insert key with value
    std::unique_ptr<TreapNode> insert_(std::unique_ptr<TreapNode> t, double key, double weight) {
        std::unique_ptr<TreapNode> l, m, r;
        split(std::move(t), key, l, m, r);
        if (m) {
            m->count += 1;
            m->node_weight += weight;
            update(m);
            return merge(merge(std::move(l), std::move(m)), std::move(r));
        }
        return merge(merge(std::move(l), std::make_unique<TreapNode>(key, weight)), std::move(r));
    }

    // erase one occurrence of key
    std::unique_ptr<TreapNode> erase_(std::unique_ptr<TreapNode> t, double key, double weight) {
        std::unique_ptr<TreapNode> l, m, r;
        split(std::move(t), key, l, m, r);
        if (m) {
            m->count -= 1;
            m->node_weight -= weight;
            update(m);
            if (m->count == 0) m = nullptr;
        }
        return merge(merge(std::move(l), std::move(m)), std::move(r));
    }
    
    // sum of keys <= x
    double prefix_value_sum_(std::unique_ptr<TreapNode>& t, double x) {
        std::unique_ptr<TreapNode> l, r;
        split(std::move(t), x, l, r);
        double res = get_subtree_value(l);
        t = merge(std::move(l), std::move(r));
        return res;
    }

    // weight sum of keys <= x
    double prefix_weight_sum_(std::unique_ptr<TreapNode>& t, double x) {
        std::unique_ptr<TreapNode> l, r;
        split(std::move(t), x, l, r);
        double res = get_subtree_weight(l);
        t = merge(std::move(l), std::move(r));
        return res;
    }

public:
    // constructor
    Treap() : root(nullptr) {}

    // insert key
    void insert(double key, double weight) {
        root = insert_(std::move(root), key, weight);
    }

    // erase one occurrence of key
    void erase(double key, double weight) {
        root = erase_(std::move(root), key, weight);
    }

    // get prefix value sum of keys <= x
    double prefix_value_sum(double x) {
        return prefix_value_sum_(root, x);
    }

    // get prefix weight sum of keys <= x
    double prefix_weight_sum(double x) {
        return prefix_weight_sum_(root, x);
    }
};

END_NAMESPACE