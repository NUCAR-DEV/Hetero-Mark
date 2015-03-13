#include "sw.h"

#include <memory>
#include <cmath>
#include <stdio.h>
#include <unistd.h>

#define ENABLE_PROFILE 1

#if ENABLE_PROFILE
#define clEnqueueNDRangeKernel clTimeNDRangeKernel
#endif

void advance_spinner()
{
    static char bars[] = { '/', '-', '\\', '|' };
    static int nbars = sizeof(bars) / sizeof(char);
    static int pos = 0;

    printf("%c\b", bars[pos]);
    fflush(stdout);
    pos = (pos + 1) % nbars;
}

ShallowWater::ShallowWater(unsigned _M, unsigned _N)
        :
        M(_M),
        N(_N),
        M_LEN(_M+1),
        N_LEN(_N+1),
        ITMAX(250),
        dt(90.),
        tdt(dt),
        dx(100000.),
        dy(100000.),
        fsdx(4. / dx),
        fsdy(4. / dy),
        a(1000000.),
        alpha(.001),
        el(N * dx),
        pi(4. * atanf(1.)),
        tpi(pi + pi),
        di(tpi / M),
        dj(tpi / N),
        pcf(pi * pi * a * a / (el * el))
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);

        InitKernel();
        InitBuffer();
}

ShallowWater::~ShallowWater()
{
        FreeKernel();
        FreeBuffer();
}

void ShallowWater::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("sw_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        if (err != CL_SUCCESS)
        {
            char buf[0x10000];
            clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_LOG,
                                  0x10000,
                                  buf,
                                  NULL);
            printf("\n%s\n", buf);
            exit(-1);
        }

        // Create kernels
        kernel_sw_init_psi_p = clCreateKernel(program, "sw_init_psi_p", &err);
        checkOpenCLErrors(err, "Failed to create kernel_sw_init_psi_p");

        kernel_sw_init_velocities = clCreateKernel(program, "sw_init_velocities", &err);
        checkOpenCLErrors(err, "Failed to create kernel_sw_init_velocities");

        kernel_sw_compute0 = clCreateKernel(program, "sw_compute0", &err);
        checkOpenCLErrors(err, "Failed to create sw_compute0");

        kernel_sw_periodic_update0 = clCreateKernel(program, "sw_periodic_update0", &err);
        checkOpenCLErrors(err, "Failed to create sw_periodic_update0");
}

void ShallowWater::InitBuffer()
{
        size_t sizeInBytes = sizeof(double) * M_LEN * N_LEN;



        // Fine grain buffers
        u_curr = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        u_next = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        v_curr = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        v_next = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        p_curr = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        p_next = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        cu     = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        cv     = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        z      = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        h      = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
        psi    = (double *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeInBytes, 0);
}

void ShallowWater::FreeKernel()
{
        
}

void ShallowWater::FreeBuffer()
{
       clSVMFree(context, u_curr);
       clSVMFree(context, u_next);
       clSVMFree(context, v_curr);
       clSVMFree(context, v_next);
       clSVMFree(context, p_curr);
       clSVMFree(context, p_next);
       clSVMFree(context, cu);
       clSVMFree(context, cv);
       clSVMFree(context, z);
       clSVMFree(context, h);
       clSVMFree(context, psi);
}

void ShallowWater::Init()
{
        InitPsiP();
        InitVelocities();
}

void ShallowWater::InitPsiP()
{
        cl_int err;

        err  = clSetKernelArg(kernel_sw_init_psi_p, 0, sizeof(double), (void *)&a);
        err |= clSetKernelArg(kernel_sw_init_psi_p, 1, sizeof(double), (void *)&di);
        err |= clSetKernelArg(kernel_sw_init_psi_p, 2, sizeof(double), (void *)&dj);
        err |= clSetKernelArg(kernel_sw_init_psi_p, 3, sizeof(double), (void *)&pcf);
        err |= clSetKernelArg(kernel_sw_init_psi_p, 4, sizeof(unsigned), (void *)&M_LEN);
        err |= clSetKernelArg(kernel_sw_init_psi_p, 5, sizeof(unsigned), (void *)&M_LEN);
        err |= clSetKernelArgSVMPointer(kernel_sw_init_psi_p, 6, (void *)p_curr);
        err |= clSetKernelArgSVMPointer(kernel_sw_init_psi_p, 7, (void *)psi);
        checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_init_psi_p");

        const size_t globalSize[2] = {M_LEN, N_LEN};
        const size_t localSize[2] = {16, 16};

        err = clEnqueueNDRangeKernel(cmdQueue, 
                                    kernel_sw_init_psi_p, 
                                    2, 
                                    NULL, 
                                    globalSize, 
                                    localSize, 
                                    0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_init_psi_p");
        
        advance_spinner();
}

void ShallowWater::InitVelocities()
{
        cl_int err;

        const size_t globalSize[2] = {M, N};
        const size_t localSize[2] = {16, 16};

        err  = clSetKernelArg(kernel_sw_init_velocities, 0, sizeof(double), (void *)&dx);
        err |= clSetKernelArg(kernel_sw_init_velocities, 1, sizeof(double), (void *)&dy);
        err |= clSetKernelArg(kernel_sw_init_velocities, 2, sizeof(unsigned), (void *)&M);
        err |= clSetKernelArg(kernel_sw_init_velocities, 3, sizeof(unsigned), (void *)&N);
        err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities, 4, (void *)psi);
        err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities, 5, (void *)u_curr);
        err |= clSetKernelArgSVMPointer(kernel_sw_init_velocities, 6, (void *)v_curr);
        checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_init_velocities");

        err = clEnqueueNDRangeKernel(cmdQueue, 
                                    kernel_sw_init_velocities, 
                                    2, 
                                    NULL, 
                                    globalSize, 
                                    localSize, 
                                    0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_init_psi_p");

        advance_spinner();
}

void ShallowWater::Compute0()
{
        cl_int err;

        const size_t globalSize[2] = {M_LEN, N_LEN};
        const size_t localSize[2] = {16, 16};

        err  = clSetKernelArg(kernel_sw_compute0, 0, sizeof(double), (void *)&fsdx);
        err |= clSetKernelArg(kernel_sw_compute0, 1, sizeof(double), (void *)&fsdy);
        err |= clSetKernelArg(kernel_sw_compute0, 2, sizeof(unsigned), (void *)&M_LEN);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 3, (void *)u_curr);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 4, (void *)v_curr);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 5, (void *)p_curr);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 6, (void *)cu);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 7, (void *)cv);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 8, (void *)z);
        err |= clSetKernelArgSVMPointer(kernel_sw_compute0, 9, (void *)h);
        checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_compute0");

        err = clEnqueueNDRangeKernel(cmdQueue, 
                                    kernel_sw_compute0, 
                                    2, 
                                    NULL, 
                                    globalSize, 
                                    localSize, 
                                    0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_compute0");

        advance_spinner();
}

void ShallowWater::PeriodicUpdate0()
{
        cl_int err;

        const size_t globalSize[2] = {M, N};
        const size_t localSize[2] = {16, 16};

        err  = clSetKernelArg(kernel_sw_periodic_update0, 0, sizeof(unsigned), (void *)&M);
        err |= clSetKernelArg(kernel_sw_periodic_update0, 1, sizeof(unsigned), (void *)&N);
        err |= clSetKernelArg(kernel_sw_periodic_update0, 2, sizeof(unsigned), (void *)&M_LEN);
        err |= clSetKernelArgSVMPointer(kernel_sw_periodic_update0, 3, (void *)cu);
        err |= clSetKernelArgSVMPointer(kernel_sw_periodic_update0, 4, (void *)cv);
        err |= clSetKernelArgSVMPointer(kernel_sw_periodic_update0, 5, (void *)z);
        err |= clSetKernelArgSVMPointer(kernel_sw_periodic_update0, 6, (void *)h);
        checkOpenCLErrors(err, "Failed to set kernel args: kernel_sw_periodic_update0");

        err = clEnqueueNDRangeKernel(cmdQueue, 
                                    kernel_sw_periodic_update0, 
                                    2, 
                                    NULL, 
                                    globalSize, 
                                    localSize, 
                                    0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueNDRangeKernel kernel_sw_periodic_update0");
        
        advance_spinner();
}

void ShallowWater::Compute1()
{
        tdts8 = tdt / 8.;
        tdtsdx = tdt / dx;
        tdtsdy = tdt / dy;



}

void ShallowWater::PeriodicUpdate1()
{

}

void ShallowWater::TimeSmooth()
{

}

void ShallowWater::Run()
{
        std::cout << "Running... ";

        Init();

        for (int i = 0; i < ITMAX; ++i)
        {
                Compute0();
                PeriodicUpdate0();
                Compute1();
                PeriodicUpdate1();
                TimeSmooth();
        }

        std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
        std::unique_ptr<ShallowWater> sw(new ShallowWater());
    
        sw->Run();

        return 0;
}