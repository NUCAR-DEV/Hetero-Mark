#ifndef PARIIR_H
#define PARIIR_H

#define ROWS 256  // num of parallel subfilters

class ParIIR
{
	int len;
	int channels;
	float c;

	float *nsec;
	float *dsec;

	// CPU output for comparison
	float *cpu_y;

	// Memory objects
	float *X;
	float *gpu_Y;

	//-----------------------------------
	// Initialize functions
	//-----------------------------------
	void Init();
	void InitParam();
	void InitBuffers();

	//----------------------------------
	// Clear functions
	//-----------------------------------
	void CleanUp();
	void CleanUpBuffers();

	// Run kernels
	void multichannel_pariir();

	// check the results
	void compare();

public:
	ParIIR(int len);
	~ParIIR();

	void Run();
};

#endif
