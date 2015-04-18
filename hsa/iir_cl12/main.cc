#include "IirFilter.h"

int main() 
{
	IirFilter iir = IirFilter();
	iir.Init();
	iir.Run();
	iir.Verify();
}
