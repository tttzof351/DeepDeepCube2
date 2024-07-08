#ifndef CUBE2_GAME_H
#define CUBE2_GAME_H

#include <iostream>
#include <vector>

#include <omp.h>

class Cube3Game {
    public:
    
    int space_size = -1;
    int action_size = -1;
    int* actions = nullptr;
    bool debug = true;

    Cube3Game() {    
        ;
    }

    ~Cube3Game() {
        if (actions != nullptr) {
            delete[] actions;
            actions = nullptr;
        }
    }

    void set_actions(
        int action_size, // Must be 12
        int space_size, // Must be 54
        double* external_actions
    ) {
        if (debug) {
            std::cout << "set_actions; action_size: " << action_size << "; space_size: " << space_size << std::endl;
        }

        this->space_size = space_size;
        this->action_size = action_size;

        this->actions = new int[action_size * space_size];
        
        // #pragma omp simd
        for (int i = 0; i < action_size * space_size; i++) {
            this->actions[i] = int(external_actions[i]);
        }

        // if (debug) {
        //     cout << "action[0] in c++: ";
        //     for (int i = 0; i < this->space_size; i++) {
        //         cout << this->actions[i] << " ";
        //     }
        //     cout << endl;
        // }
    }

    bool is_goal(std::vector<float>& state) {
        for (int i = 0; i < this->space_size; i++) {
            if (int(state[i]) != i) {
                return false;
            }            
        }
        return true;
    }

    bool is_goal(double* raw_input) {
        for (int i = 0; i < this->space_size; i++) {
            if (int(raw_input[i]) != i) {
                return false;
            }
        }

        return true;
    }

    void apply_action(
        std::vector<float>& in_state,
        std::vector<float>& out_state,
        int action
    ) {
        for (int i = 0; i < this->space_size; i++) {
            out_state[i] = int(in_state[
                int(this->actions[action * this->space_size + i])
            ]);
            // out_state[i] = this->space_size - i;
        }
    }
};


#endif
