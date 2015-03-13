#ifndef SHALLOW_WATER_H
#define SHALLOW_WATER_H

#include <clUtil.h>

using namespace clHelper;

class ShallowWater
{
        clRuntime *runtime;
        clFile    *file;

        cl_platform_id   platform;
        cl_device_id     device;
        cl_context       context;
        cl_command_queue cmdQueue;

        cl_program       program;
        cl_kernel        kernel_sw_init_psi_p;
        cl_kernel        kernel_sw_init_velocities;
        cl_kernel        kernel_sw_compute0;
        cl_kernel        kernel_sw_periodic_update0;
        cl_kernel        kernel_sw_compute1;

        // Size
        unsigned M;
        unsigned N;
        unsigned M_LEN;
        unsigned N_LEN;
        unsigned ITMAX;

        // Params
        double dt,tdt,dx,dy,a,alpha,el,pi;
        double tpi,di,dj,pcf;
        double tdts8,tdtsdx,tdtsdy,fsdx,fsdy;

        // SVM buffers
        double *u_curr, *u_next;
        double *v_curr, *v_next;
        double *p_curr, *p_next;
        double *cu, *cv;
        double *z, *h, *psi;

        void InitKernel();
        void InitBuffer();

        void FreeKernel();
        void FreeBuffer();

        void Init();
        void InitPsiP();
        void InitVelocities();

        void Compute0();
        void PeriodicUpdate0();
        void Compute1();
        void PeriodicUpdate1();
        void TimeSmooth();

public:
        ShallowWater(unsigned _M = 2048, unsigned _N = 2048);
        ~ShallowWater();

        void Run();
        
};

#endif
