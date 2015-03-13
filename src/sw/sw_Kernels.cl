__kernel void sw_init_psi_p(         const double a,
                                     const double di,
                                     const double dj,
                                     const double pcf,
                                     const unsigned M_LEN,
                                     const unsigned N_LEN,
                            __global       double *p,
                            __global       double *psi)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if(x <= M_LEN && y <= N_LEN)
        {
                psi[y * M_LEN + x] = a * sin((x + .5) * di) * sin((y + .5) * dj);
                p[y * M_LEN + x] = pcf * (cos(2. * (x) * di) + cos(2. * (y) * dj)) + 50000.;
        }
}

__kernel void sw_init_velocities(         const double dx,
                                          const double dy,
                                          const unsigned M,
                                          const unsigned N,
                                 __global const double *psi,
                                 __global       double *u,
                                 __global       double *v)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if(x <= M && y <= N)
        {
                u[(y+1) * M + x] = -(psi[(y + 1) * M + x + 1] - psi[(y + 1) * M + x]) / dy;
                v[y * M + x + 1] =  (psi[(y + 1) * M + x + 1] - psi[y * M + x + 1]) / dx;
        }
}

__kernel void sw_compute0(        const double fsdx,
                                  const double fsdy,
                                  const unsigned M_LEN,

                         __global const double *u,
                         __global const double *v,
                         __global const double *p,

                         __global       double *cu,
                         __global       double *cv,
                         __global       double *z,
                         __global       double *h)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        cu[(y+1)*M_LEN+x] = 0.5 * (p[(y+1)*M_LEN+x] + p[y*M_LEN+x]) * u[(y+1)*M_LEN+x];
        cv[y*M_LEN+x+1] = 0.5 * (p[y*M_LEN+x+1] + p[y*M_LEN+x]) * v[y*M_LEN+x+1];
        z[(y+1)*M_LEN+x+1] = (fsdx * (v[(y+1)*M_LEN+x+1] - v[y*M_LEN+x+1]) - fsdy * (u[(y+1)*M_LEN+x+1] - u[(y+1)*M_LEN+x])) 
                / (p[y*M_LEN+x] + p[(y+1)*M_LEN+x] + p[(y+1)*M_LEN+x+1] + p[y*M_LEN+x+1]);
        h[y*M_LEN+x] = p[y*M_LEN+x] + 0.25 * (u[(y+1)*M_LEN+x] * u[(y+1)*M_LEN+x] 
                                            + u[y*M_LEN+x] * u[y*M_LEN+x]
                                            + v[y*M_LEN+x+1] * v[y*M_LEN+x+1] 
                                            + v[y*M_LEN+x] * v[y*M_LEN+x]);         
}

__kernel void sw_periodic_update0(         const unsigned M,
                                           const unsigned N,
                                           const unsigned M_LEN,
                                  __global       double *cu,
                                  __global       double *cv,
                                  __global       double *z,
                                  __global       double *h)
{
        
        int x = get_global_id(0);
        int y = get_global_id(1);

        if(x < N)
        {
                cu[x] = cu[M*M_LEN + x];
                cv[M*M_LEN + x + 1] = cv[x + 1];
                z[x + 1] = z[M*M_LEN + x + 1];
                h[M*M_LEN + x] = h[x];
        }

        if(y < M)
        {
                cu[(y + 1)*M_LEN + N] = cu[(y + 1)*M_LEN];
                cv[y*M_LEN] = cv[y*M_LEN + N];
                z[(y + 1)*M_LEN] = z[(y + 1)*M_LEN + N];
                h[y*M_LEN + N] = h[y*M_LEN];
        }

        cu[N] = cu[M*M_LEN];
        cv[M*M_LEN] = cv[N];
        z[0] = z[M*M_LEN + N];
        h[M*M_LEN + N] = h[0];
}

