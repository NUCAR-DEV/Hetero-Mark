#include "pagerank.h"
#include <memory>

int main(int argc, char const *argv[])
{
	if (argc < 3) {
		std::cout << "Usage: pagerank input_matrix input_vector" << endl;
		exit(-1);
	}
	std::unique_ptr<PageRank> pr(new PageRank(argv[1], argv[2]));
	pr->ReadCsrMatrix();
	pr->ReadDenseVector();
	return 0;
}
