#ifndef COMMON_BENCHMARK_H
#define COMMON_BENCHMARK_H

/**
 * A benchmark is a program that tests performance. 
 * This class is an abstract class, which provides some basic infrastructure.
 */
class Benchmark 
{

public:
	/**
	 * Init setup
	 */
	virtual void Init() = 0;

	/**
	 * Run the benchmark
	 */
	virtual void Run() = 0;

	/**
	 * Verify the execution result
	 */
	virtual void Verify() = 0;

};

#endif
