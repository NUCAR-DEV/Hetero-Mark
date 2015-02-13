#ifndef CL_PROFILER_H
#define CL_PROFILER_H

#include <sys/time.h>
#include <map>
#include <memory>

namespace clHelper
{

class clProfiler
{
        // Instance of the singleton
        static std::unique_ptr<clProfiler> instance;

        // Private constructor for singleton
        clProfiler();

        // Contains profiling data
        std::map<std::string, double> profilingData;

        // String length
        size_t strLen;

public:

        ~clProfiler();

        // Get singleton
        static clProfiler *getInstance();

        // Get number of record
        int getNumRecord() const { return profilingData.size(); };

        // Dump kernel profiling time
        void getExecTime(std::string name = "");

        // Add profiling info
        void addExecTime(std::string name, double execTime);

        // Set max string length
        void setStringLen(size_t strLen) { this->strLen = strLen; }
};

// Singleton instance
std::unique_ptr<clProfiler> clProfiler::instance;

clProfiler *clProfiler::getInstance()
{
        // Instance already exists
        if (instance.get())
                return instance.get();
        
        // Create instance
        instance.reset(new clProfiler());
        return instance.get();
}

clProfiler::clProfiler()
     :
     strLen(16)
{

}

clProfiler::~clProfiler()
{

}

void clProfiler::getExecTime(std::string name)
{
        if (name != "")
        {
                std::string sampleName = name;
                sampleName.resize(strLen, ' ');
                if(profilingData.find(sampleName) != profilingData.end())
                        std::cout << sampleName << " = " << profilingData[sampleName] 
                                  << " ms" << std::endl;
        }
        else
        {
                double totalTime = 0.0f;
                std::cout << "Profiler info" << std::endl;
                for(auto elem : profilingData)
                {
                        std::cout << "\t" << elem.first << " = " 
                                  << elem.second << " ms" << std::endl;
                        totalTime += elem.second;
                }
                std::cout << "Profiler total time = " << totalTime << " ms" << std::endl;

        }
}

void clProfiler::addExecTime(std::string name, double execTime)
{
        std::string sampleName = name;
        sampleName.resize(strLen, ' ');
        profilingData[sampleName] += execTime;
}

double time_stamp()
{
        struct timeval t;
        if(gettimeofday(&t, 0) != 0)
          exit(-1);
        return t.tv_sec + t.tv_usec/1e6;
}

// Enqueue and profile a kernel
cl_int clProfileNDRangeKernel(cl_command_queue cmdQ,
                              cl_kernel        kernel,
                              cl_uint          wd,
                              const size_t *   glbOs,
                              const size_t *   glbSz,
                              const size_t *   locSz,
                              cl_uint          numEvt,
                              const cl_event * evtLst,
                              cl_event *       evt)
{
        cl_int   err;
        cl_int   enqueueErr;
        cl_event perfEvent;
        cl_command_queue_properties cmdQProp;

        // Enable profiling of command queue
        // err = clSetCommandQueueProperty(cmdQ, CL_QUEUE_PROFILING_ENABLE, true, NULL);
        // checkOpenCLErrors(err, "Failed to enable profiling on command queue");

        // Enqueue kernel
        enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0, NULL, &perfEvent);
        checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");
        clWaitForEvents(1, &perfEvent);

        // Get profiling information
        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        cl_double execTimeMs = (cl_double)(end - start)*(cl_double)(1e-06); 

        // Get kernel name
        char kernelName[1024];
        err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char), (void *)kernelName, NULL);

        clProfiler *prof = clProfiler::getInstance();
        prof->addExecTime(kernelName, execTimeMs);
        
        // printf
        // printf("Kernel %s costs %f ms\n", kernelName, execTimeMs);

        return enqueueErr;
}

cl_int clTimeNDRangeKernel(cl_command_queue cmdQ,
                           cl_kernel        kernel,
                           cl_uint          wd,
                           const size_t *   glbOs,
                           const size_t *   glbSz,
                           const size_t *   locSz,
                           cl_uint          numEvt,
                           const cl_event * evtLst,
                           cl_event *       evt)
{
        cl_int   err;
        cl_int   enqueueErr;

        clFinish(cmdQ);
        
        // Enqueue kernel
        double start = time_stamp();
        enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0, NULL, NULL);
        clFinish(cmdQ);
        double end = time_stamp();
        checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");

        double execTimeMs = (double)(end - start); 

        // Get kernel name
        char kernelName[1024];
        err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char), (void *)kernelName, NULL);

        clProfiler *prof = clProfiler::getInstance();
        prof->addExecTime(kernelName, execTimeMs);

        // printf
        // printf("Kernel %s costs %f ms\n", kernelName, execTimeMs);

        return enqueueErr;
}

void DumpProfilingInfo()
{
        clProfiler *prf = clProfiler::getInstance();

        if (prf->getNumRecord())
            prf->getExecTime();
}

}

#endif
