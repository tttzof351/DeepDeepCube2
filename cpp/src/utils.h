#ifndef UTILS_H
#define UTILS_H

#include <ctime>

#include "wyhash.h"

using namespace std;

int hash_seed = 42;
uint64_t _wyp_s[4];

void init_wyhash() {
    make_secret(time(NULL), _wyp_s);
}

uint64_t w_hash(const void *data, size_t len) {
    return wyhash(data, len, hash_seed, _wyp_s);
}

#endif