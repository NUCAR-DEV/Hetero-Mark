#ifndef COMMON_TIMEKEEPER_H
#define COMMON_TIMEKEEPER_H

#include <unordered_map>
#include <memory>

/**
 * Time keeper is an service that measures and keeps execution time.
 * Time keeper assumes the execution is single thread
 */
class TimeKeeper
{

	// The global only instance of time keeper
	static TimeKeeper *instance;

	// Private constructor
	TimeKeeper();

	// The timer kept
	std::unordered_map<std::string, double> timers;

	// The time when the timmer start
	std::unique_ptr<double> start_time_in_second;

	// Accumulate time on a specific timer
	void AccumulateTime(const char *timer, double time_in_second);

public:

	/**
	 * Get the global only instance
	 */
	static TimeKeeper *getInstance();

	/**
	 * Start timer
	 */
	void BeginTimer();

	/**
	 * End timer
	 */
	void EndTimer(std::initializer_list<const char *> catagories);

	/**
	 * Dump summary information
	 */
	void Summarize();

};

#endif
