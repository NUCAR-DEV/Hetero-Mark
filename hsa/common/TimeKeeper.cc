#include <sys/time.h>

#include "TimeKeeper.h"

TimeKeeper *TimeKeeper::instance;


TimeKeeper::TimeKeeper() 
{
	start_time_in_second.reset(nullptr);
}


TimeKeeper *TimeKeeper::getInstance()
{
	if (!instance)
	{
		instance = new TimeKeeper();
	}
	return instance;
}


void TimeKeeper::BeginTimer()
{
	// Check if the timmer already started
	if (start_time_in_second.get()) 
	{
		printf("Timer already started.\n");
		exit(1);
	}

	// Get new time
	struct timeval t;
	if (gettimeofday(&t, 0) != 0)
	{
		printf("Error starting timer\n");
		exit(1);
	}
	start_time_in_second.reset(new double(t.tv_sec + t.tv_usec * 1e-6));
}


void TimeKeeper::EndTimer(std::initializer_list<const char *> catagories)
{
	// Get end time
	struct timeval t;
	if (gettimeofday(&t, 0) != 0)
	{
		printf("Error stoppint timer\n");
		exit(1);
	}

	// Check if the timer started
	if (!start_time_in_second.get())
	{
		printf("Timer have not started\n");
		exit(1);
	}

	// Calculate time passed
	double end_time_in_second = t.tv_sec + t.tv_usec * 1e-6;
	double time_passed_in_second = end_time_in_second - 
		*(start_time_in_second.get());

	// Reset the start time
	start_time_in_second.reset(nullptr);

	// Traverse all catagories
	for (auto catagory : catagories)
	{
		AccumulateTime(catagory, time_passed_in_second);
	}

	// Accumulate total time
	AccumulateTime("Total execution", time_passed_in_second);
}


void TimeKeeper::AccumulateTime(const char *catagory, double time_in_second)
{
	auto timer = timers.find(catagory);
		
	// Create catagory on demand
	if (timer == timers.end())
	{
		timers.emplace(catagory, 0);
		timer = timers.find(catagory);
	}

	// Increase time
	timer->second += time_in_second;
}


void TimeKeeper::Summarize()
{
	for (auto timer : timers)
	{
		printf("\t%s time: %f\n", timer.first.c_str(), timer.second);
	}
}
