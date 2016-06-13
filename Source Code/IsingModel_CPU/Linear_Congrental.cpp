// needed for "lcrng.hpp" for class methods to be callable on gpu
// if nvcc is used __CUDACC__ is defined
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <iostream>
#include "lcrng.hpp"
#include <stdio.h>

#define NUMBER_OF_BLOCKS 5
#define THREADS_PER_BLOCK 2
#define GENERATOR_LENGTH 5
#define GRID_SIZE 16


int main() {
	int number_of_blocks = NUMBER_OF_BLOCKS;
	int number_rngs = number_of_blocks * THREADS_PER_BLOCK;
	int number_rnds = number_rngs * GENERATOR_LENGTH;
	
	// create lcrng for seed vector
	Lcrng_cpu *get_seeds = new Lcrng_cpu(1);
	
	// calculate seeds for the lcrngs -> seeds
	unsigned int *seeds = new unsigned int[number_rngs];

	//initializes elements of seeds[]
	for (int i=0; i<number_rngs; i++) {
		seeds[i] = get_seeds->get_next_rnd();
	}
	delete get_seeds;
	
	//print all seeds
	for (int i=0; i<number_rngs; i++) {
		// printf("seed %i: %ld\n",i ,long(seeds[i]));
		std::cout << "seed " << i << ": " << seeds[i] << std::endl;
	}  
	
	// calculate all random numbers, store -> random_numbers
	// indexing: (thread_number + block_number * THREADS_PER_BLOCK) * GENERATOR_LENGTH + generator_element_i
	float *random_numbers = new float[number_rnds];

	for(int block_number=0; block_number<number_of_blocks; block_number++) {
		for(int thread_number=0; thread_number<THREADS_PER_BLOCK; thread_number++) {
			// create lcrng
			unsigned int seed_value = seeds[thread_number + block_number * THREADS_PER_BLOCK];
			Lcrng_gpu *lcrng = new Lcrng_gpu(seed_value);
			
			for(int i=0; i<GENERATOR_LENGTH; i++) {
				int rnd_pos_index = (thread_number + block_number * THREADS_PER_BLOCK) * GENERATOR_LENGTH + i;
				// printf("block: %d, thread: %d, seed: %ld, rnd_pos_ptr: %p\n", block_number, thread_number, (long) seeds[thread_number + block_number * THREADS_PER_BLOCK], (void *) rnd_pos_ptr); 
				random_numbers[rnd_pos_index] = lcrng->get_next_rnd();
				std::cout << lcrng->get_next_rnd() << std::endl;
			}
			
			delete lcrng;
		}
	}
	
	// print all random numbers to file
	FILE *f;
	f = fopen("random_numbers_cpu.dat", "w");
	fprintf(f, "{\n");
	int i;
	for (i=0; i<number_rnds; i++) {
		fprintf(f, "\t\"%i\": %f", i ,(double)random_numbers[i]);
		if (i<number_rnds-1) {
			fprintf(f, ",");
		}
		fprintf(f, "\n");
	}
	fprintf(f, "}");
	fclose(f);  

#ifdef __CUDACC__
	// device arrays
	float *d_random_numbers; 
	float *d_seeds;
	
	// allocate space on device
	cudaMalloc((void **)&d_seeds, number_rngs * sizeof(unsigned int));
	cudaMalloc((void **)&d_random_numbers, number_rnds * sizeof(float));
	
	// copy seeds to device
	cudaMemcpy(d_seeds, seeds, number_rngs * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	
	void lcrng_parallel (d_random_numbers, d_seeds) {
		int seed_index = (threadIdx.x + blockIdx.x * blockDim.x);
		seed = d_seeds[seed_index];
		Lcrng_gpu *generate_rnd = new Lcrng_gpu(seed);
		for(int i=0; i<GENERATOR_LENGTH; i++) {
			index = threadIdx.x + blockIdx.x * blockDim.x) * GENERATOR_LENGTH;
			d_random_number[index] = generate_rnd->get_next_rnd();
		}
	}
*/
	
	// launch lcrng() kernel on GPU
	lcrng_parrallel<<<number_of_blocks,THREADS_PER_BLOCK>>>(d_random_numbers, d_seeds);
	
	// copy results random numbers to host
	cudaMemcpy(random_numbers, d_random_numbers, number_rnds * sizeof(float), cudaMemcpyDeviceToHost);
	
	// cleanup
	cudaFree(d_seeds);
	cudaFree(d_random_numbers);
#endif

/*
// initialize 2D grid of spins
	short *d_spin_grid;
	int number_of_spins = GRID_SIZE * GRID_SIZE;

// allocate space on device
	cudaMalloc((void **)&d_spint_grid, number_of_spins * sizeof(short));
// */
	return 0;
}
