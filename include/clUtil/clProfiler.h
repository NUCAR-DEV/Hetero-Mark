#ifndef CL_PROFILER_H
#define CL_PROFILER_H

#include <sys/time.h>
#include <map>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace clHelper
{

class clProfilerMeta
{
        std::string name;
        double totalTime;

        int limit;

        std::vector<std::unique_ptr<std::pair<double, double>>> timeTable;

public:
        clProfilerMeta(std::string nm);
        ~clProfilerMeta();

        /// Getters
        const std::string getName() const { return name; }

        const double getTotalTime() const { return totalTime; }

        /// Record a profiling information
        void insert(double st, double ed);

        /// Operator \c << invoking the function Dump on an output stream
        friend std::ostream &operator<<(std::ostream &os,
                const clProfilerMeta &clProfMeta)
        {
                clProfMeta.Dump(os);
                return os;
        }
        
        void Dump(std::ostream &os) const;
};

clProfilerMeta::clProfilerMeta(std::string nm)
        :name(nm),
         totalTime(0.0f),
         limit(10)
{

}

clProfilerMeta::~clProfilerMeta()
{

}

void clProfilerMeta::insert(double st, double ed)
{
        timeTable.emplace_back(new std::pair<double, double>(st, ed));
        totalTime += (ed - st);
}

void clProfilerMeta::Dump(std::ostream &os) const
{
        os << "\t" << name << " : " << totalTime << " ms" << std::endl;

        // Only dump detailed data when size is not too large
        int count = 0;
        for(auto &elem : timeTable)
        {
                double st = elem.get()->first;
                double ed = elem.get()->second;
                double lt = ed - st;
                os << "\t\t" << std::setprecision(32) << st << " " << 
                  std::setprecision(32) << ed << ": " << std::setprecision(20) << 
                  lt << " ms, (#" << count << ")" << std::endl;
                
                count++;
                if(count > limit) 
                {
                       os << "\t\t......" << timeTable.size() - count 
                         << " more, " << timeTable.size() << " records in total\n";
                       break;
                } 


        }                
}

class clProfiler
{
        // Instance of the singleton
        static std::unique_ptr<clProfiler> instance;

        // Private constructor for singleton
        clProfiler();

        // Contains profiling data
        std::vector<std::unique_ptr<clProfilerMeta>> profilingData;

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
        void addExecTime(std::string name, double st, double ed);

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
        // Profiling info at the end of program execution
        if (getNumRecord())
            getExecTime();
}

void clProfiler::getExecTime(std::string name)
{
        if (name != "")
        {
                std::string sampleName = name;
                sampleName.resize(strLen, ' ');
                for(auto &meta : profilingData)
                {
                        if (meta->getName() == sampleName)
                        {
                                std::cout << *meta;
                        }
                }
        }
        else
        {
                double totalTime = 0.0f;
                std::cout << "Kernel Profiler info" << std::endl;
                for(auto &meta : profilingData)
                {
                        std::cout << *meta;
                        totalTime += meta->getTotalTime();
                }
                std::cout << "Profiler total time = " << std::setprecision(8) 
                  << totalTime*1e-6 << " s/ " 
                  << totalTime << " ms" <<std::endl;
        }
}

void clProfiler::addExecTime(std::string name, double st, double ed)
{
        std::string sampleName = name;
        sampleName.resize(strLen, ' ');

        // Check if already in the list
        for(auto &elem : profilingData)
        {
                if (elem->getName() == sampleName)
                {
                        elem->insert(st, ed);
                        return;
                }
        }

        // Create if not in the list
        profilingData.emplace_back(new clProfilerMeta(sampleName));
        profilingData.back()->insert(st, ed);

}

double time_stamp_ms()
{
        struct timeval t;
        if(gettimeofday(&t, 0) != 0)
          exit(-1);
        return t.tv_sec*1e6 + t.tv_usec;
}

double time_stamp()
{
        struct timeval t;
        if(gettimeofday(&t, 0) != 0)
          exit(-1);
        return t.tv_sec + t.tv_usec*1e-6;
}

// Enqueue and profile a kernel
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
        // prof->addExecTime(kernelName, start/1e6, end/1e6);
        
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
        double start = time_stamp_ms();
        enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0, NULL, NULL);
        clFinish(cmdQ);
        double end = time_stamp_ms();
        checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");

        double execTimeMs = (double)(end - start); 

        // Get kernel name
        char kernelName[1024];
        err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char), (void *)kernelName, NULL);
        
        clProfiler *prof = clProfiler::getInstance();
        prof->addExecTime(kernelName, start, end);

        // printf
        // printf("Kernel %s costs %f ms\n", kernelName, execTimeMs);

        return enqueueErr;
}

}

#endif
