typedef struct GpuPartitioner {
  int n_data;
  int current;
  int *worklist;
  int *position;
} GpuPartitioner;

__device__ inline GpuPartitioner gpu_partitioner_create(int n_data,
                                                        int *worklist,
                                                        int *position) {
  GpuPartitioner p;
  p.n_data = n_data;
  p.worklist = worklist;
  p.position = position;
  return p;
}

__device__ inline int gpu_initialize(GpuPartitioner *p) {
  p->current = atomicAdd_system(p->worklist, 1);
  return p->current;
}

__device__ inline bool gpu_more(const GpuPartitioner *p) {
  return (p->current < p->n_data);
}

__device__ inline int gpu_increment(GpuPartitioner *p) {
  p->current = atomicAdd_system(p->worklist, 1);
  return p->current;
}
