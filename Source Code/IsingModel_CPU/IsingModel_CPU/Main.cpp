/*
  #####################################
	Ising Model - CPU Implementation
  #####################################

	Checkerboard Metropolis Algorithm
	For Cubic Lattices

  #####################################

	Ising Model Hamiltonian: H = <sum_i> H_i
	H_i = - (J <sum_j> S_j + H) S_i
   ++++++++++++++++++++++++++++++++++++
	Units:
	Magnetic Field B: [B] = [H / µ_B] = T
	Coupling Energy: [J] = eV
	Temperature: [T] = K
	Lattice Constant: [a] = m
	Spin Quantum Number S: [S] = 1

  #####################################
*/

// +++++++++++++++++++++++++
// Libraries
// +++++++++++++++++++++++++

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdexcept>

// +++++++++++++++++++++++++
// Preprocessor Definitions
// +++++++++++++++++++++++++

#define NOERR		0	// no error
#define ERR			1	// general error
#define UP			1	// spin state: up
#define DOWN		-1	// spin state: down
#define INIT_RAND	1	// initial state: random
#define INIT_UP		2	// initial state: all states UP (+1)
#define INIT_DOWN	3	// initial state: all states DOWN (-1)
#define INIT_ALT	4	// initial state: alternating states
#define PI			3.141592653589	// pi
#define K_B			8.6173303e-5	// Boltzmann constant [k_B] = eV / K
#define MU_B		5.7883818e-5	// Bohr Magneton [µ_B] = eV / T
#define MU_0		2.0133545e-25	// vacuum permeability [µ_0] = T^2m^3 / eV

using namespace std;


// ##############################
// SYSTEM SETTINGS
// ##############################

// +++++++++++++++++++++++++
// Static Definitions
// +++++++++++++++++++++++++

#define	FLOATTYPE	double		// single/double precision
#define DIM			3			// dimension of lattice
#define INIT_STATE	INIT_ALT	// initial state
#define ITERATIONS	1000		// main grid iteration count
#define COUPLING	1e-3		// coupling energy [eV] (J>0 ferromagnetism, J<0 antiferromagnetism)
#define LATTICE		2.8665e-10	// lattice constant [m]


// +++++++++++++++++++++++++
// Functional Definitions
// +++++++++++++++++++++++++

#define SIZE		30			// edge length of lattice (cubic probe)
#define TEMP		30			// temperature [K] (homogenous within probe)
#define FIELD		.5			// external field [T] (homogenous within probe), H_ext = g_s µ_B B_ext

// Lattice Size
int setLatticeSize(int (&latticeSize)[DIM], int &siteCount) {
	siteCount = 1;
	if (DIM != 1 && DIM != 2 && DIM != 3) return ERR;
	for (int i = 0; i < DIM; i++) {
		latticeSize[i] = 2 * ((SIZE + 1) / 2);	// integer division => even edge lengths
		siteCount *= latticeSize[i];
	}

	return NOERR;
}

// Temperature Evolution
FLOATTYPE getTemp() {
	return TEMP;
}
/*
FLOATTYPE getTemp(int iterationStep) {
	return T(t);
}
*/

// Field Evolution
FLOATTYPE getFieldEnergy() {
	return MU_B * FIELD;
}
/*
FLOATTYPE getField(int iterationStep, int x, int y, int z) {
	return H(t, x, y, z);
}
*/


// ##############################
// FUNCTIONS
// ##############################

// +++++++++++++++++++++++++
// Random Number Generators
// +++++++++++++++++++++++++

// Normalised Random Number Generator
// return: random number between 0 and 1
FLOATTYPE randNorm() {
	FLOATTYPE result = float(rand()) / RAND_MAX;
	return result;
}

// Random State Generator
// return: random number -1 or 1
int randState() {
	int result = 2 * (rand() % 2) - 1;
	return result;
}


// +++++++++++++++++++++++++
// Array Operations
// +++++++++++++++++++++++++

// Print Integer Array (optional as float)
// array: Array
// size: Size of Array
// length: One-Dimensional Array Size (Matrix Output for Hypercubic Arrays)
template <typename T>
int printArray(T array[], int size, int length = -1, bool spinRepresentation = false, bool floatRepresentation = false) {
	if (length == -1) length = size;
	int length2 = length * length; if (length2 > size) length2 = size;
	try {
		if (spinRepresentation) {		// UP = x, DOWN = '
			for (int i = 0; i < size; i++) {
				if (i % length2 == 0) printf("[\n");
				if (i % length == 0) printf(" {");
				if (int(array[i]) == 1) printf(" x ");
				else if (int(array[i]) == -1) printf(" ' ");
				if (i % length == length - 1) printf("}\n");
				if (i % length2 == length2 - 1) printf("]\n");
			}
		}
		else if (floatRepresentation) {
			for (int i = 0; i < size; i++) {
				if (i % length2 == 0) printf("[\n");
				if (i % length == 0) printf(" {");
				printf(" %2.2f ", int(array[i]));
				if (i % length == length - 1) printf("}\n");
				if (i % length2 == length2 - 1) printf("]\n");
			}
		}
		else {
			for (int i = 0; i < size; i++) {
				if (i % length2 == 0) printf("[\n");
				if (i % length == 0) printf(" {");
				printf("  %2d  ", int(array[i]));
				if (i % length == length - 1) printf("}\n");
				if (i % length2 == length2 - 1) printf("]\n");
			}
		}
	}
	catch (...) {
		return ERR;
	}
	return NOERR;
}

// Total of Array
// array: Array
// dim: Size of Array
// return: Sum of Array Elements
template <typename T>
T totalArray(T array[], int dim) {
	T result = 0;
	try {
		for (int i = 0; i < dim; i++) result += array[i];
	}
	catch (...) {
		fprintf(stderr, "Error: totalArray(array[], dim)");
	}
	return result;
}


// +++++++++++++++++++++++++
// Lattice Operations
// +++++++++++++++++++++++++

// Get Lattice Coordinate
// (x, y, z): Cartesian Coordinates
// return: Lattice Sorted Coordinate (Array Index)
int mapCoordinateLattice(int x, int y = 0, int z = 0) {
	return x + SIZE * y + SIZE * SIZE * z;
}

// Get Cartesian Coordinate
// index: Lattice Sorted Coordinate
// n: Cartesian Coordinate Selection (1 == x, 2 == y, 3 == z)
// return: Respective 1/2/3 (x/y/z) Cartesian Coordinate
int mapCoordinateCartesian(int index, int n) {
	return (index % int(pow(SIZE, n))) / int(pow(SIZE, n - 1));	// (using integer division)
}

// Initialise Lattice State
// array: Lattice
// siteCount: Total Count of Lattice Sites
int initialiseLatticeState(FLOATTYPE *array, int siteCount) {
	try {
		switch (INIT_STATE) {
		case INIT_RAND:	for (int i = 0; i < siteCount; i++) { array[i] = randState(); } break;
		case INIT_UP:	for (int i = 0; i < siteCount; i++) { array[i] = UP; } break;
		case INIT_DOWN:	for (int i = 0; i < siteCount; i++) { array[i] = DOWN; } break;
		case INIT_ALT:	for (int i = 0; i < siteCount; i++) { array[i] = (2 * (mapCoordinateCartesian(i, 1) % 2) - 1) * (2 * (mapCoordinateCartesian(i, 2) % 2) - 1) * (2 * (mapCoordinateCartesian(i, 3) % 2) - 1); } break; // equivalent: (-1)^(x+y+z)
		default: return ERR;
		}
	}
	catch (...) {
		return ERR;
	}

	return NOERR;
}


// +++++++++++++++++++++++++
// Energy Operations
// +++++++++++++++++++++++++

// Boltzmann Factor exp(-E / kT)
FLOATTYPE boltzmannFactor(FLOATTYPE energy, FLOATTYPE temperature) {
	return exp(-energy / K_B / temperature);
}

// Local Energy of Local Spin as given by Ising Model
// latticeState: Lattice Spin States
// field: External Field Energy
// (x, y, z): Cartesian Coordinate of Local Spin
int updateLocalEnergy(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, FLOATTYPE fieldEnergy, int x, int y = 0, int z = 0) {
	int status = 0;
	FLOATTYPE result = 0;

	if (x > 0) result += latticeState[mapCoordinateLattice(x - 1, y, z)];
	if (x < SIZE - 1) result += latticeState[mapCoordinateLattice(x + 1, y, z)];
#if DIM > 1
	if (y > 0) result += latticeState[mapCoordinateLattice(x, y - 1, z)];
	if (y < SIZE - 1) result += latticeState[mapCoordinateLattice(x, y + 1, z)];
#endif
#if DIM > 2
	if (z > 0) result += latticeState[mapCoordinateLattice(x, y, z - 1)];
	if (z < SIZE - 1) result += latticeState[mapCoordinateLattice(x, y, z + 1)];
#endif

	latticeEnergy[mapCoordinateLattice(x, y, z)] = -latticeState[mapCoordinateLattice(x, y, z)] * (COUPLING * result + fieldEnergy);
	return status;
}

// Local Spin Flip
// latticeState: Lattice Spin States
// latticeEnergy: Lattice Energies
// temperature: Temperature of System
// (x, y, z): Cartesian Coordinate of Local Spin
// return: true if spin was flipped
bool flipLocalSpin(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, FLOATTYPE temperature, int x, int y = 0, int z = 0) {
	// dE = E(-S_i) - E(S_i) = -2 E(S_i) => exp(-2 E(S_i) / kT)
	if (randNorm() < boltzmannFactor(-2 * latticeEnergy[mapCoordinateLattice(x, y, z)], temperature)) {
		latticeState[mapCoordinateLattice(x, y, z)] = -latticeState[mapCoordinateLattice(x, y, z)];
		return true;
	}
	else {
		return false;
	}
}

// Magnetisation (homogenous distribution)
// latticeState: Lattice Spin States
// latticeSize: Dimensions of Lattice
// return: Magnetisation [T]
FLOATTYPE getMagnetisation(FLOATTYPE *latticeState, int *latticeSize, int siteCount = pow(SIZE, DIM)) {
	FLOATTYPE spinTotal = 0;
#if DIM == 1
	for (int x = 0; x < latticeSize[0]; x++) {
		spinTotal += latticeState[mapCoordinateLattice(x)];
	}
#endif
#if DIM == 2
	for (int x = 0; x < latticeSize[0]; x++) {
		for (int y = 0; y < latticeSize[1]; y++) {
			spinTotal += latticeState[mapCoordinateLattice(x, y)];
		}
	}
#endif
#if DIM == 3
	for (int x = 0; x < latticeSize[0]; x++) {
		for (int y = 0; y < latticeSize[1]; y++) {
			for (int z = 0; z < latticeSize[2]; z++) {
				spinTotal += latticeState[mapCoordinateLattice(x, y, z)];
			}
		}
	}
#endif
	return MU_0 * MU_B * spinTotal / LATTICE / LATTICE / LATTICE / siteCount;	// M = µ_0 µ_B dS / a^3
}

// Initialise Local Energies
int initialiseLatticeEnergy(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, int *latticeSize) {
	int status = 0;
	try {
#if DIM == 1
		for (int x = 0; x < latticeSize[0]; x++) {
			status += updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x);
		}
#endif
#if DIM == 2
		for (int y = 0; y < latticeSize[1]; y++) {
			for (int x = 0; x < latticeSize[0]; x++) {
				status += updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x, y);
			}
		}
#endif
#if DIM == 3
		for (int z = 0; z < latticeSize[2]; z++) {
			for (int y = 0; y < latticeSize[1]; y++) {
				for (int x = 0; x < latticeSize[0]; x++) {
					status += updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x, y, z);
				}
			}
		}
#endif
	}
	catch (...) {
		fprintf(stderr, "Error: initialiseLatticeEnergy(latticeState, latticeEnergy, latticeSize");
		status++;
	}
	return status;
}


// +++++++++++++++++++++++++
// Iteration Operations
// +++++++++++++++++++++++++

// Iteration over Thread
int iterateThread(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, int *latticeSize, bool subgrid, int x, int y = 0, int z = 0) {
	int status = 0;
	FLOATTYPE cache = 0;

	try{
#if DIM == 1
		updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x);
		// sync
		flipLocalSpin(latticeState, latticeEnergy, getTemp(), x);
#endif
#if DIM == 2
		for (int dx = 0; dx < 2; dx++) {
			for (int dy = 0; dy < 2; dy++) {
				if ((dx + dy) % 2 == int(subgrid)) updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x + dx, y + dy);
			}
		}
		// sync
		for (int dx = 0; dx < 2; dx++) {
			for (int dy = 0; dy < 2; dy++) {
				if ((dx + dy) % 2 == int(subgrid)) flipLocalSpin(latticeState, latticeEnergy, getTemp(), x + dx, y + dy);
			}
		}
#endif
#if DIM == 3
		for (int dx = 0; dx < 2; dx++) {
			for (int dy = 0; dy < 2; dy++) {
				for (int dz = 0; dz < 2; dz++) {
					if ((dx + dy + dz) % 2 == int(subgrid)) updateLocalEnergy(latticeState, latticeEnergy, getFieldEnergy(), x + dx, y + dy, z + dz);
				}
			}
		}
		// sync
		for (int dx = 0; dx < 2; dx++) {
			for (int dy = 0; dy < 2; dy++) {
				for (int dz = 0; dz < 2; dz++) {
					if ((dx + dy + dz) % 2 == int(subgrid)) flipLocalSpin(latticeState, latticeEnergy, getTemp(), x + dx, y + dy, z + dz);
				}
			}
		}
#endif

	}
	catch (...) {
		fprintf(stderr, "Error: iterateThread(latticeState, latticeEnergy, latticeSize, subgrid, x, y, z)");
		status++;
	}
	return status;
}

// Iteration over Block
int iterateBlock(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, int *latticeSize, bool subgrid, int y = 0, int z = 0) {
	int status = 0;
	try{
		for (int x = 0; x < latticeSize[0] / 2; x++) {
			status += iterateThread(latticeState, latticeEnergy, latticeSize, subgrid, 2 * x, y, z);
		}
	}
	catch (...) {
		fprintf(stderr, "Error: iterateBlock(latticeState, latticeEnergy, latticeSize, subgrid, y, z)");
		status++;
	}
	return status;
}

// Iteration over Subgrid
int iterateSubGrid(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, int *latticeSize, bool subgrid) {
	int status = 0;
	// call block iterations
	try {
#if DIM == 1
		status += iterateBlock(latticeState, latticeEnergy, latticeSize, subgrid);
#else
		for (int y = 0; y < latticeSize[1] / 2; y++) {
#if DIM == 2
			status += iterateBlock(latticeState, latticeEnergy, latticeSize, subgrid, 2 * y);
#endif
#if DIM == 3
			for (int z = 0; z < latticeSize[2] / 2; z++) {
				status += iterateBlock(latticeState, latticeEnergy, latticeSize, subgrid, 2 * y, 2 * z);
			}
#endif
		}
#endif
	}
	catch (...) {
		fprintf(stderr, "Error: iterateMainGrid(latticeState, latticeEnergy, latticeSize, subgrid)");
		status++;
	}

	return status;
}

// Iteration over Main Grid
int iterateMainGrid(FLOATTYPE *latticeState, FLOATTYPE *latticeEnergy, int *latticeSize, int iterationCount) {
	int status = NOERR;
	for (int i = 0; i < iterationCount; i++) {
		status += iterateSubGrid(latticeState, latticeEnergy, latticeSize, true);
		// sync
		status += iterateSubGrid(latticeState, latticeEnergy, latticeSize, false);
	}
	return status;
}



// ##############################
// MAIN
// ##############################

int main() {
	// Timer
	double clockStart = clock();

	// Initialisations
	srand(time(NULL));
	int errStatus = NOERR;

	// System Parameters
	int latticeSize[DIM], siteCount; errStatus = setLatticeSize(latticeSize, siteCount);
	if (errStatus != NOERR) goto End;
	
	FLOATTYPE *latticeState = new FLOATTYPE[siteCount];	// latticeState[z][y][x] = latticeState[z * SIZE * SIZE + y * SIZE + x]
	errStatus = initialiseLatticeState(latticeState, siteCount); if (errStatus != NOERR) goto End;

	FLOATTYPE *latticeEnergy = new FLOATTYPE[siteCount]; // latticeEnergy[z][y][x] as retrieved from Ising model
	errStatus = initialiseLatticeEnergy(latticeState, latticeEnergy, latticeSize); if (errStatus != NOERR) goto End;


	// DEBUG
	printf("INITIAL State:\nMagnetisation M = %fT\n", getMagnetisation(latticeState, latticeSize, siteCount));
	printArray(latticeState, siteCount, SIZE, true);
	//printArray(latticeEnergy, siteCount, SIZE, false, true);
	printf("\n\n");

	// Metropolis Algorithm
	iterateMainGrid(latticeState, latticeEnergy, latticeSize, ITERATIONS);

	// DEBUG
	printf("FINAL State:\nMagnetisation M = %fT\n", getMagnetisation(latticeState, latticeSize, siteCount));
	printArray(latticeState, siteCount, SIZE, true);
	//printArray(latticeEnergy, siteCount, SIZE, false, true);
	printf("\n\n");


	// Timer
	double elapsedTime = (double(clock()) - clockStart) / CLOCKS_PER_SEC;
	printf("Program Run Time: %.0fms", 1000 * elapsedTime);

End:
	if (errStatus != NOERR) fprintf(stderr, "Error");

	delete[] latticeState;
	delete[] latticeEnergy;

	_fgetchar();
	return NOERR;
}