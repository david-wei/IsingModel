# include <climits>

// Base class linear congruental
class Lcrng 
{
	public:	
		Lcrng() {}
		~Lcrng() {}
	protected:
		// parameters for linear congruental rng:
		// rnd_(i+1) = (a * rnd_i + c) % m
		unsigned int a;
		unsigned int m;
		unsigned int c;
		unsigned int current_rnd;	
};

// Derived class for random vector for seed values on cpu only
// method get_next_rnd() returns a unsigned int because these numbers
// will be used for Lcrng_gpu as seed values
class Lcrng_cpu: public Lcrng
{
	public:
		Lcrng_cpu (unsigned int seed_in, 
		unsigned int a_in=16807, unsigned int c_in=0, 
		unsigned int m_in=UINT_MAX) {
			current_rnd = seed_in;
			a = a_in;
			m = m_in;
			c = c_in;
		}
		
		~Lcrng_cpu () {}
		
		unsigned int get_next_rnd() {
			// updates current_rnd and returns updated number float between 0 and 1
			current_rnd = (a * current_rnd) % m;
			return current_rnd;
		}
};

// Derived class for calculation of random numbers con be used on gpu and cpu
// One Lcrng_gpu can be passed to each thread on gpu and then be
// used to calculate new rnds for spin update
class Lcrng_gpu: public Lcrng
{
	public:
		// CUDA_CALLABLE_MEMBER gets defined as __host__ __device__ 
		// by precompiler if nvcc is used
		CUDA_CALLABLE_MEMBER Lcrng_gpu(unsigned int seed_in, 
		unsigned int a_in=1664525, unsigned int c_in=1013904223, 
		unsigned int m_in=UINT_MAX) {
			current_rnd = seed_in;
			a = a_in;
			m = m_in;
			c = c_in;
		}
		
		CUDA_CALLABLE_MEMBER ~Lcrng_gpu() {}
		
		CUDA_CALLABLE_MEMBER float get_next_rnd() {
			// updates current_rnd and returns updated number float between 0 and 1
			current_rnd = (a * current_rnd) % m;
			return (float(current_rnd) / float(m));
		}
};
