#ifndef CATBOOST_PARALLEL_A_STAR_H
#define CATBOOST_PARALLEL_A_STAR_H

#include <omp.h>
#include <vector>
#include <queue>
#include <map>

#include "utils.h"
#include "nodes.h"
#include "cube3_game.h"
#include "bloom_filter.hpp"

// #include "../../assets/models/catboost_cube3.cpp"

class CatboostParallelAStar {
    public:

    int limit_size = -1;
    
    int parallel_size = 10;
    int open_max_size = 10000;
    float alpha = 0.9;
    // int open_optimum_size = 10000;

    bool debug = true;


    NodeMinMaxBloom open;
    NodeMinMaxBloom close;

    // NodeQueue open;
    // NodeQueue close;

    CatboostParallelAStar(
        Cube3Game& game,
        int limit_size,
        double* init_state_raw,
        bool debug,
        int parallel_size = 10,
        int open_max_size = 10000,
        float alpha = 0.9
    ) {
        this->limit_size = limit_size;
        this->debug = debug;
        
        this->parallel_size = parallel_size;
        this->open_max_size = open_max_size;
        this->alpha = alpha;

        Node* root_node = new Node(game.space_size, this->alpha);
        
        for (int i = 0; i < game.space_size; i++) {
            root_node->state->at(i) = int(init_state_raw[i]);
        }
        
        root_node->h = ApplyCatboostModel(*root_node->state);
        root_node->reset_hash();
        root_node->reset_f();

        this->open.insert(root_node);
    }

    ~CatboostParallelAStar() {
        for (auto it = open.hashes.begin(); it != open.hashes.end(); it++) {
            delete it->second;
        }

        for (auto it = close.hashes.begin(); it != close.hashes.end(); it++) {
            delete it->second;
        }
    }

    Node* search(Cube3Game& game) {
        if (debug) {
            cout << "search, size open: " << open.size() << endl;
        }
        
        Node* child_nodes[game.action_size];
        int global_i = 0;
        int prev_global_i = 0;

        auto start = std::chrono::high_resolution_clock::now();

        while (open.size() > 0) {        
            std::vector<Node*> best_parallel_q;
            
            for (int i = 0; i < parallel_size; i++) {
                if (open.size() > 0) {
                    Node* best_node = this->open.pop_min_element();    
                    best_parallel_q.push_back(best_node);
                } else {
                    break;
                }
            }
            
            Node* target = nullptr;
            std::vector<Node*>* candidates_to_open = new std::vector<Node*>[parallel_size];
            
            // std::cout << "start parallel size:" << best_parallel_q.size() << std::endl;

            #pragma omp parallel for
            for (int i = 0; i < best_parallel_q.size(); i++) {
                // cout << "parallel i:" << i << std::endl;
                Node* best_node = best_parallel_q[i];
                Node* child_nodes[game.action_size];

                //Initialization childs
                #pragma omp parallel for
                for (int action = 0; action < game.action_size; action++) {
                    Node* child = new Node(game.space_size, this->alpha);
                    child->action = action;

                    game.apply_action(
                        *(best_node->state), // in
                        *(child->state), // out
                        action
                    );

                    child->h = ApplyCatboostModel(*(child->state));
                    child->g = best_node->g + 1;

                    child->parent = best_node;                
                    child->reset_hash();
                    child->reset_f();

                    child_nodes[action] = child;
                }

                // cout << "end child i:" << i << std::endl;
                for (int action = 0; action < game.action_size; action++) {
                    Node* child = child_nodes[action];
                    bool is_goal = game.is_goal(*(child->state));

                    if (is_goal) {
                        //For prevent memory leak
                        // cout << "FOUND GOAL" << endl;
                        // for (int j = action + 1; j < game.action_size; j++) {
                        //     cout << "Try to remove: " << j << endl;
                        //     delete child_nodes[j];
                        // }

                        target = child;
                    } else if (close.is_contains(child)) {
                        delete child;
                        continue;
                    } else if (open.is_contains(child)) {
                        //TODO: Need implementation
                        delete child;

                        continue;
                    } else {
                        // this->open.insert(child);
                        candidates_to_open[i].push_back(child);
                    }
                }
                // TODO: Move outside for
                // this->close.insert(best_node);

                #pragma omp atomic
                global_i += 1;
            }

            // cout << "end parallel: global_i " << global_i << std::endl;

            for (int i = 0; i < best_parallel_q.size(); i++) {
                std::vector<Node*>& append_to_opens = candidates_to_open[i];
                for (int j = 0; j < append_to_opens.size(); j++) {
                    Node* node = append_to_opens[j];
                    if (open.is_contains(node)) {
                        delete node;
                    } else {
                        open.insert(append_to_opens[j]);
                    }
                }
            }

            for (int i = 0; i < best_parallel_q.size(); i++) {
                Node* best_node = best_parallel_q[i];
                close.insert(best_node);
                best_node->clear_state();
            }
            delete[] candidates_to_open;

            if (target != nullptr) {
                // cout << "I found target!" << endl;
                return target;
            }

            if (debug && (global_i - prev_global_i) > 1000) {
                auto end = std::chrono::high_resolution_clock::now();
                
                cout << "size close: " 
                << this->close.size() 
                << "; size open: "
                << this->open.size()                
                << "; Duration: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()  
                << " ms"
                << endl; 

                start = end;

                prev_global_i = global_i;
            }

            if (open.size() > this->open_max_size) {
                while (open.size() > open_max_size) {
                    Node* node = open.pop_max_element();
                    delete node;
                }
                // open.reset_bloom();
            }

            if (global_i > this->limit_size) {
                return nullptr;
            }            
        }

        return nullptr;
    }

    int close_size() {
        return close.size();
    }
};


#endif