#ifndef NODES_H
#define NODES_H

#include <vector>
#include <queue>
#include <map>
#include <set>

#include "utils.h"
#include "bloom_filter.hpp"
#include "MinMaxHeap.hpp"

using namespace std;

class Node {
    public:

    int space_size = -1;
    vector<float>* state = nullptr; //TODO: Move to stack
    uint64_t state_hash = -1;

    float g = 0;
    float h = 0;
    float f = 0;

    float alpha = 1.0;

    Node* parent = nullptr;
    int action = -1;

    Node(
        int space_size
    ) {
        this->space_size = space_size;
        this->state = new vector<float>(space_size);
    }

    Node(
        int space_size,
        float alpha
    ) : Node (space_size) {
        this->alpha = alpha;
    }    

    ~Node() {
        if (this->state != nullptr) {
            delete this->state;
            this->state = nullptr;
        }
    }

    void clear_state() {
        delete this->state;
        this->state = nullptr;
    }

    void reset_hash() {
        this->state_hash = w_hash(
            (void*)&this->state->at(0), 
            this->space_size * sizeof(float)
        );
    }

    void reset_f() {
        f = alpha * g + h;
    }

    vector<Node*> get_path() {
        Node* node = this;
        vector<Node*> path;

        while (node != nullptr) {
            path.push_back(node); // TODO: push_front ? 
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());

        return move(path);
    }
};

struct CompareNode {
    bool operator()(const Node* n1, const Node* n2) {
        return n1->f > n2->f;
    }
};

struct CompareNode2 {
    bool operator()(Node* n1, const Node* n2) const {
        return n1->f > n2->f;
    }
};

struct CompareNode3 {
    bool operator()(Node* n1, const Node* n2) const {
        return n1->f < n2->f;
    }
};


class NodeQueue {
    public:

    std::priority_queue <Node*, vector<Node*>, CompareNode> queue;
    std::map<uint64_t, Node*> hashes;

    NodeQueue() {
        ;
    }

    ~NodeQueue() {
        ;
    }

    void insert(Node* node) {
        this->queue.push(node);
        this->hashes[node->state_hash] = node;
    }

    Node* pop_min_element() {
        Node* node = this->queue.top();
        this->queue.pop();

        this->hashes.erase(node->state_hash);

        return node;
    }

    int size() {
        return this->queue.size();
    }

    bool is_contains(Node* node) {
        return this->hashes.find(node->state_hash) != this->hashes.end();
    }
};

class NodeSet {
    public:

    std::set<Node*, CompareNode2> queue;
    std::map<uint64_t, Node*> hashes;

    NodeSet() {
        ;
    }

    ~NodeSet() {
        ;
    }

    void insert(Node* node) {
        this->queue.insert(node);
        this->hashes[node->state_hash] = node;
    }

    Node* pop_min_element() {
        Node* node = *this->queue.rbegin();
        queue.erase(node);

        this->hashes.erase(node->state_hash);

        return (Node*)node;
    }

    int size() {
        return this->queue.size();
    }

    bool is_contains(Node* node) {
        return this->hashes.find(node->state_hash) != this->hashes.end();
    }
};

class NodeMinMax {
    public:

    minmax::MinMaxHeap<Node*, std::vector<Node*>, CompareNode3> heap;
    std::map<uint64_t, Node*> hashes;

    NodeMinMax() {
    }

    ~NodeMinMax() {
    }

    void insert(Node* node) {
        this->heap.push(node);
        this->hashes[node->state_hash] = node;

    }

    Node* pop_min_element() {
        Node* node = heap.popMin();
        this->hashes.erase(node->state_hash);

        return (Node*)node;
    }

    int size() {
        return this->heap.size();
    }

    bool is_contains(Node* node) {
        return this->hashes.find(node->state_hash) != this->hashes.end();
    }

    Node* pop_max_element() {
        Node* node = heap.popMax();
        this->hashes.erase(node->state_hash);

        return node;
    }
};

class NodeMinMaxBloom {
    public:

    bloom_parameters parameters;
    minmax::MinMaxHeap<Node*, std::vector<Node*>, CompareNode3> heap;
    std::map<uint64_t, Node*> hashes;
    bloom_filter* filter = nullptr;

    NodeMinMaxBloom() {
        parameters.projected_element_count = 1000000;
        parameters.false_positive_probability = 0.1;
        parameters.random_seed = 0xA5A5A5A5;
        parameters.compute_optimal_parameters();
        
        filter = new bloom_filter(parameters);
    }

    ~NodeMinMaxBloom() {
        delete filter;
    }

    void insert(Node* node) {
        this->heap.push(node);
        this->hashes[node->state_hash] = node;
        this->filter->insert(node->state_hash);   
    }

    Node* pop_min_element() {
        Node* node = heap.popMin();
        this->hashes.erase(node->state_hash);

        return (Node*)node;
    }

    int size() {
        return this->heap.size();
    }

    bool is_contains(Node* node) {
        // bool is_contains_in_bloom = this->filter->contains(node->state_hash);
        // if (~is_contains_in_bloom) {
        //     return false;
        // }

        return this->hashes.find(node->state_hash) != this->hashes.end();
    }

    Node* pop_max_element() {
        Node* node = heap.popMax();
        this->hashes.erase(node->state_hash);

        return node;
    }

    void reset_bloom() {
        delete filter;
        filter = new bloom_filter(parameters);

        for (auto it = hashes.begin(); it != hashes.end(); it++) {
            filter->insert(it->second->state_hash);
        }
    } 
};

#endif