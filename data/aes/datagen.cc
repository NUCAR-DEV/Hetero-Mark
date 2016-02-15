#include <cstdio>
#include <ctime>
#include <cstdlib>

void DumpHelp() {
  printf("Format: datagen number_of_bytes");
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    DumpHelp(); 
    exit(1);
  }

  unsigned long long size = atoi(argv[1]);
 
  srand(time(NULL));
  for (unsigned long long i = 0; i < size; i++) {
    unsigned char byte = rand() % 16;
    printf("%02x", byte);
  }
  
}
