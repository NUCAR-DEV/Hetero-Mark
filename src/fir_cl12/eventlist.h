#ifndef _EVENTLIST_
#define _EVENTLIST_

#include <iostream>
#include <vector>

#include <sstream>
#include <string>
#include <CL/cl.h>

//! A Better Event Handling Class
class EventList
{	

public:

	// Create a type for the time to avoid confusion
	typedef cl_ulong cl_time;

	// Structure for user event information (we can't use OpenCL user
	// events because they don't support profiling information)
	struct _cl_user_event {
		cl_ulong queued;
		cl_ulong submitted;
		cl_ulong start;
		cl_ulong end;
	};
	typedef _cl_user_event* cl_user_event;

	// Constructor
	EventList(cl_context context, cl_command_queue commandQueue,
			cl_device_id device, bool free_events=true);

	// Destructor
	~EventList();

	// Add an OpenCL event to the event list
	void add(cl_event event, const char* name=NULL,
			const char* type=NULL);

	// Add a user event to the event list
	void add(cl_user_event event, const char* name=NULL,
			const char* type=NULL);

	// Command to create user event (designed similar to OpenCL API)
	static cl_user_event clCreateUserEvent();

	// Command to set user event status (records time when status is set)
	static cl_int clSetUserEventStatus(cl_user_event, cl_int execution_status);

	// Writes event information to file
	void dumpEvents(char* path);

	// Prints the events to the screen
	void printEvents();

	// Queries the GPU and CPU clocks at (roughly) the same time
	void resetClocks();

private:	

	// Structure for holding information for each OpenCL event
	struct _event_triple {
		cl_event event;
		const char* name;
		const char* type;
	};
	typedef _event_triple* event_triple;

	// Structure for holding information for each user event
	struct _user_event_triple {
		cl_user_event event;
		const char* name;
		const char* type;
	};
	typedef _user_event_triple* user_event_triple;

	// OpenCL command queue for the application
	cl_command_queue commandQueue;

	// OpenCL context for the application
	cl_context context;

	// Timer offset for the CPU
	cl_ulong cpu_timer_start;

	// Creates a filename for the event dump based on the current time
	char* createFilenameWithTimestamp();

	// OpenCL device for the application
	cl_device_id device;

	// List of OpenCL events
	std::vector<event_triple> event_list;

	// Boolean for whether we should free the events for the user
	bool free_events;

	// Function to get the current system (CPU) time
	static cl_time getCurrentTime();

	// Function to get a profiling value from an OpenCL event
	cl_time getEventValue(cl_event event, cl_profiling_info param_name);

	// Function to get a profiling value from a user event
	cl_time getUserEventValue(EventList::cl_user_event event,
			cl_profiling_info param_name);

	// Timer offset for the GPU
	cl_ulong gpu_timer_start;

	// Performs as safe (hopefully quick) string copy
	static char* strCopy(const char* origStr);

	// List of user events
	std::vector<user_event_triple> user_event_list;

};

#endif 
