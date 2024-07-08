#ifndef HEURISTIC_A_STAR_H
#define HEURISTIC_A_STAR_H

#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

#include "utils.h"
#include "nodes.h"
#include "cube3_game.h"

class HeuristicAStar {
    public:
    
    py::function& heuristic;
    int limit_size = -1;
    bool debug = false;    
    NodeQueue open;
    NodeQueue close;

    HeuristicAStar(
        py::function& heuristic,
        Cube3Game& game,
        int limit_size,
        double* init_state_raw,
        bool debug
    ): heuristic(heuristic) {
        this->limit_size = limit_size;
        this->debug = debug;

        Node* root_node = new Node(game.space_size);
        
        //TODO: Memcopy
        for (int i = 0; i < game.space_size; i++) {
            root_node->state->at(i) = int(init_state_raw[i]);
        }
        
        py::object result = heuristic(*root_node->state);
        vector<float> hs = result.cast<vector<float>>();
        root_node->h = hs[0];

        root_node->reset_hash();
        root_node->reset_f();

        this->open.insert(root_node);
    }

    ~HeuristicAStar() {
        for (auto it = open.hashes.begin(); it != open.hashes.end(); it++) {
            delete it->second;
        }

        for (auto it = close.hashes.begin(); it != close.hashes.end(); it++) {
            delete it->second;
        }
    }

    Node* search(Cube3Game& game) {
        if (this->debug) {
            cout << "search, size open: " << this->open.size() << endl;
        }
        
        Node* child_nodes[game.action_size];
        std::vector<float> all_child_states(game.action_size * game.space_size);

        int global_i = 0;

        auto start = std::chrono::high_resolution_clock::now();
        
        while (this->open.size() > 0) {
            Node* best_node = this->open.pop_min_element();

            //Initialization childs
            #pragma omp parallel for
            for (int action = 0; action < game.action_size; action++) {
                Node* child = new Node(game.space_size);
                child->action = action;

                game.apply_action(
                    *(best_node->state), // in
                    *(child->state), // out
                    action
                );

                std::memcpy(
                    &all_child_states[action * game.space_size], //dst
                    &child->state->at(0), //src
                    game.space_size * sizeof(float)
                );                

                child->reset_hash();

                child->g = best_node->g + 1;
                child->parent = best_node;                
                child_nodes[action] = child;
            }
            
            py::object result = heuristic(all_child_states);
            vector<float> hs = result.cast<vector<float>>();
            
            #pragma omp parallel for
            for (int action = 0; action < game.action_size; action++) {
                Node* child = child_nodes[action];
                child->h = hs[action];
                child->reset_f();
            }

            for (int action = 0; action < game.action_size; action++) {
                Node* child = child_nodes[action];
                bool is_goal = game.is_goal(*(child->state));

                if (is_goal) {
                    //For prevent memory leak
                    for (int j = action + 1; j < game.action_size; j++) {
                        delete child_nodes[j];
                    }

                    return child; 
                                
                } else if (this->close.is_contains(child)) {
                    delete child;
                    continue;
                } else if (this->open.is_contains(child)) {
                    //TODO: Need implementation
                    delete child;

                    continue;
                } else {
                    this->open.insert(child);
                }
            }

            this->close.insert(best_node);            
            
            global_i += 1;
            if (debug && global_i % 1000 == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                
                cout << "size close: " 
                << this->close.size() 
                << "; Duration: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()  
                << " ms"
                << endl; 

                start = end;
            }

            if (global_i > this->limit_size) {
                return nullptr;
            }
        }

        return nullptr;
    }
};


#endif
