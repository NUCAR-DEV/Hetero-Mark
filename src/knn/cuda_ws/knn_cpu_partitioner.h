#include <atomic>

typedef struct CpuPartitioner {
   int n_data;
   int current;
   std::atomic_int *worklist;
} CpuPartitioner;

inline CpuPartitioner cpu_partitioner_create(int n_data, std::atomic_int *worklist)
{
  CpuPartitioner p;
  p.n_data = n_data;
  p.worklist = worklist;
  return p; 
}


inline int  cpu_initializer(CpuPartitioner *p) {
  
  p->current = p->worklist->fetch_add(1);
  return p->current;
}

inline bool cpu_more(const CpuPartitioner *p) {
  return (p->current < p->n_data);
}


inline int cpu_increment(CpuPartitioner *p)
{
  p->current = p->worklist->fetch_add(1);
  return p->current;
}

