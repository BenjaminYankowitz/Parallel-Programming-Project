#include <stdio.h>
#include <stdint.h>


double genrand_real1(void);
uint32_t genrand_int_n(uint32_t n); // gen number from [0, n]


void init_by_array(unsigned long init_key[], int key_length);