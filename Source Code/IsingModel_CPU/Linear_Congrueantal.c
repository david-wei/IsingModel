#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define NUMBER_OF_BLOCKS 5
#define THREADS_PER_BLOCK 16
#define GENERATOR_LENGTH 500
#define GRID_SIZE 16

void lcrng_seed(unsigned int* rnd_numbers, int rng_length, unsigned int seed) {
	unsigned int x = seed;
	unsigned int a = 16807;
	unsigned int m = UINT_MAX;
	int i;
	for (i=0; i<rng_length; i++) {
		x = (a * x ) % m;
		*rnd_numbers = x;
		rnd_numbers++;
	}
}

/*
__global__ void lcrng(float *rnd_numbers, unsigned int *seed) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * GENERATOR_LENGTH;
	float *rnd_pos_ptr = rnd_numbers + index
	unsigned int x = seed;
	unsigned int a = 1664525;
	unsigned int c = 1013904223;
	unsigned int m = UINT_MAX;
	int i;
	for (i=0; i<GENERATOR_LENGTH; i++) {
		x = (a * x + c) % m;
		*rnd_numbers = x;
		rnd_numbers++;
	}
}
*/ 

void lcrng(float* rnd_numbers, int rng_length, unsigned int seed) {
	unsigned int x = seed;
	unsigned int a = 1664525;
	unsigned int c = 1013904223;
	unsigned int m = UINT_MAX;
	int i;
	for (i=0; i<rng_length; i++) {
		x = (a * x + c) % m;
		printf("%f\n", (double) ((float) x) / ((float) m));
		*rnd_numbers = ((float) x) / ((float) m);
		rnd_numbers++;
	}
}
int main() {
	int number_of_blocks = NUMBER_OF_BLOCKS;
	int number_rngs = number_of_blocks * THREADS_PER_BLOCK;
	int number_rnds = number_rngs * GENERATOR_LENGTH;
	
	// calculate seeds for the lcrngs -> seeds
	unsigned int seed = 1; // first seed for lcrng to create seeds for other rngs
	unsigned int *seeds;
	seeds = (unsigned int *) malloc(number_rngs * sizeof(unsigned int));
	if(seeds != NULL) {
		printf("\nSpeicher ist reserviert\n");
	}else {
		printf("\nKein freier Speicher vorhanden.\n");
	}
	lcrng_seed(seeds, number_rngs, seed); //initializes elements of seeds[]
	
	/* //print all seeds
	int i;
	for (i=0; i<number_rngs; i++) {
		printf("seed %i: %ld\n",i ,(long)seeds[i]);
	}  */
	
	// calculate all random numbers, store -> random_numbers
	float *random_numbers; // indexing: (thread_number + block_number * THREADS_PER_BLOCK) * GENERATOR_LENGTH + generator_element_i
	random_numbers = (float *) malloc(number_rnds * sizeof(float));
	int block_number, thread_number;
	for(block_number=0; block_number<number_of_blocks; block_number++) {
		for(thread_number=0; thread_number<THREADS_PER_BLOCK; thread_number++) {
		float *rnd_pos_ptr = random_numbers + (thread_number + block_number * THREADS_PER_BLOCK) * GENERATOR_LENGTH;
		// printf("block: %d, thread: %d, seed: %ld, rnd_pos_ptr: %p\n", block_number, thread_number, (long) seeds[thread_number + block_number * THREADS_PER_BLOCK], (void *) rnd_pos_ptr); 
		lcrng(rnd_pos_ptr, GENERATOR_LENGTH, seeds[thread_number + block_number * THREADS_PER_BLOCK]);
		}
	}
	
	// print all random numbers to file
	FILE *f;
	f = fopen("random_numbers_cpu.dat", "w");
	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}
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
	
/* cuda code for random number generation
	// device arrays
	float *d_random_numbers; 
	float *d_seeds;
	
	// allocate space on device
	cudaMalloc((void **)&d_seeds, number_rngs * sizeof(unsigned int));
	cudaMalloc((void **)&d_random_numbers, number_rnds * sizeof(float));
	
	// copy seeds to device
	cudaMemcpy(d_seeds, seeds, number_rngs * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// launch lcrng() kernel on GPU
	lcrng<<<number_of_blocks,THREADS_PER_BLOCK>>>(d_random_numbers, d_seeds);
	
	// copy results random numbers to host
	cudaMemcpy(random_numbers, d_random_numbers, number_rnds * sizeof(float), cudaMemcpyDeviceToHost);
	
	// cleanup
	cudaFree(d_seeds);
	cudaFree(d_random_numbers);
*/

//*
// initialize 2D grid of spins
	short *d_spin_grid;
	int number_of_spins = GRID_SIZE * GRID_SIZE;

// allocate space on device
	cudaMalloc((void **)&d_spint_grid, number_of_spins * sizeof(short));
// */
	free(random_numbers);
	free(seeds);
	return 0;
}
