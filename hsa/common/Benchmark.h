#ifndef COMMON_BENCHMARK_H
#define COMMON_BENCHMARK_H

#include "TimeKeeper.h"

/**
 * A benchmark is a program that tests performance. 
 * This class is an abstract class, which provides some basic infrastructure.
 */
class Benchmark 
{
protected:

	// Time Keeper
	TimeKeeper *timer;

	// Runtime helper
	HsaHelper *helper;

public:

	/**
	 * Constructor
	 */
	Benchmark()
	{
		timer = TimeKeeper::getInstance();
	}
	
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

	/**
	 * Summarize the benchmark result
	 */
	virtual void Summarize() = 0;

	/**
	 * Inject helper
	 */
	void setHelper(HsaHelper *helper) { this->helper = helper; }

	/**
	 * Inject time keeper if required
	 */
	void setTimeKeeper(TimeKeeper *timer) { this->timer = timer; }

};

#endif
