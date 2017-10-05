#include <iostream>
#include <atomic>


// Partitioner definition -----------------------------------------------------

typedef struct DynamicPartitioner {

    int n_tasks;
    int current;
    //std::atomic_int *worklist;
    int *worklist;
    int *tmp;
} DynamicPartitioner;


__device__ inline DynamicPartitioner partitioner_create(int n_tasks, float alpha ,int *worklist, int *tmp)
  {
    Partitioner p;
    p.n_tasks = n_tasks;
    p.worklist = worklist;
    p.tmp = tmp;
    
    return p;
}


inline int cpu_first(DynamicPartitioner *p) {
  p->current = p->worklist->fetch_add(1);
  return p->current;
}



__device__  int gpu_first(DynamicPartitioner *p) {

  if(threadIdx.y == 0 && threadIdx.x == 0) {
            p->tmp[0] = atomicAdd_system(p->worklist, 1);
   }
   __syncthreads();
   p->current = p->tmp[0];
   return p->current;
}


inline bool cpu_more(const DynamicPartitioner *p) {
    return (p->current < p->n_tasks);
}

__device__ bool gpu_more(const DynamicPartitioner *p) {
    return (p->current < p->n_tasks);
}

inline int cpu_next(DynamicPartitioner *p) {
    p->current = p->worklist->fetch_add(1);
    return p->current;
}



__device__ int gpu_next(DynamicPartitioner *p) {
        if(threadIdx.y == 0 && threadIdx.x == 0) {
            p->tmp[0] = atomicAdd_system(p->worklist, 1);
        }
        __syncthreads();
        p->current = p->tmp[0];
        return p->current;
}
