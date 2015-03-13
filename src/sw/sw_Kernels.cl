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

        if(x < N_LEN && y < M_LEN)
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

        if(x < N && y < M)
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

__kernel void sw_update0(         const unsigned M,
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

__kernel void sw_compute1(
                                   const double tdts8,
                                   const double tdtsdx,
                                   const double tdtsdy,
                                   const unsigned M_LEN,
                          __global const double *cu, 
                          __global const double *cv, 
                          __global const double *z, 
                          __global const double *h,
                          __global const double *u_curr, 
                          __global const double *v_curr, 
                          __global const double *p_curr, 
                          __global       double *u_next, 
                          __global       double *v_next, 
                          __global       double *p_next)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        u_next[(y+1) * M_LEN + x] = u_curr[(y+1) * M_LEN + x] + tdts8 * (z[(y+1) * M_LEN + x + 1] +
                                z[(y+1) * M_LEN + x]) * (cv[(y+1) * M_LEN + x + 1] +
                                cv[y * M_LEN + x + 1] + cv[y * M_LEN + x] + cv[(y+1) * M_LEN + x]) -
                                tdtsdx * (h[(y+1) * M_LEN + x] - h[y* M_LEN + x]);
        
        v_next[y * M_LEN + x + 1] = v_curr[y * M_LEN + x + 1] - tdts8 * (z[(y+1) * M_LEN + x + 1] +
                        z[y * M_LEN + x + 1]) * (cu[(y+1) * M_LEN + x + 1] +
                        cu[y * M_LEN + x + 1] + cu[y * M_LEN + x] + cu[y * M_LEN + x + 1]) -
                        tdtsdy * (h[y * M_LEN + x + 1] - h[y * M_LEN + x]);

        p_next[y * M_LEN + x] = p_curr[y * M_LEN + x] - tdtsdx * (cu[(y + 1) * M_LEN + x] - cu[y * M_LEN + x]) -
                        tdtsdy * (cv[y * M_LEN + x + 1] - cv[y * M_LEN + x]);
 
}

__kernel void sw_update1(         const unsigned M,
                                  const unsigned N,
                                  const unsigned M_LEN,
                         __global       double *u_next,
                         __global       double *v_next,
                         __global       double *p_next)
{
        
        int x = get_global_id(0);
        int y = get_global_id(1);

        if(x < N)
        {
                u_next[x] = u_next[M*M_LEN + x];
                v_next[M*M_LEN + x + 1] = v_next[x + 1];
                p_next[M*M_LEN + x] = p_next[x];
        }

        if(y < M)
        {
                u_next[(y + 1)*M_LEN + N] = u_next[(y + 1)*M_LEN];
                v_next[y*M_LEN] = v_next[y*M_LEN + N];
                p_next[y*M_LEN + N] = p_next[y*M_LEN];
        }

        u_next[N] = u_next[M*M_LEN];
        v_next[M*M_LEN] = v_next[N];
        p_next[M*M_LEN + N] = p_next[0];

}

__kernel void sw_time_smooth(         const unsigned M,
                                      const unsigned N,
                                      const unsigned M_LEN,
                                      const double   alpha,
                             __global       double  *u,
                             __global       double  *v,
                             __global       double  *p,
                             __global       double  *u_curr,
                             __global       double  *v_curr,
                             __global       double  *p_curr,
                             __global       double  *u_next,
                             __global       double  *v_next,
                             __global       double  *p_next)
{
        int x = get_global_id(0);
        int y = get_global_id(1);

        u_curr[y * M_LEN + x] = u[y * M_LEN] + alpha * (u_next[y * M_LEN + x] - 2. * u[y * M_LEN + x] + u_curr[y * M_LEN + x]);
        v_curr[y * M_LEN + x] = v[y * M_LEN] + alpha * (v_next[y * M_LEN + x] - 2. * v[y * M_LEN + x] + v_curr[y * M_LEN + x]);        
        p_curr[y * M_LEN + x] = p[y * M_LEN] + alpha * (p_next[y * M_LEN + x] - 2. * p[y * M_LEN + x] + p_curr[y * M_LEN + x]);
}
