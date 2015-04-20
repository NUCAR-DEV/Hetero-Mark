#include <cstring>

#include "../common/OptionParser.h"

#include "IirFilter.h"

uint32_t len = 1024;
bool is_verification_mode = false;

void ParseArgument(int argc, char **argv)
{
	OptionParser parser = OptionParser(argv[0], 
			"The IIR filter benchmark");

	// Set length argument
	parser.AddArgument(
			"Data Length", 
			"-l", "--length",
			"The length of the data to be processed", 
			"1024",
			typeid(unsigned int)
			);

	// Set verification mode argument
	parser.AddArgument(
			"Verification Mode", 
			"-v", "--verify",
			"Enter verification mode", 
			"false",
			typeid(bool)
			);	

	// Parse argument
	parser.Parse(argc, argv);

	// Get length
	len = parser.getValue<uint32_t>("Data Length");
	printf("Len: %d\n", len);

	// Get verification mode
	is_verification_mode = parser.getValue<bool>("Verification Mode");
	printf("Verification mode: %s\n", is_verification_mode?"true":"false");
}


int main(int argc, char **argv) 
{
	// Parse argument
	ParseArgument(argc, argv);

	// Setup runtime helper
	HsaHelper helper = HsaHelper();
	helper.setVerificationMode(is_verification_mode);

	// Init benchmark
	IirFilter iir = IirFilter();
	iir.setHelper(&helper);
	iir.setDataLength(len);
	iir.Init();

	// Run benchmark
	iir.Run();

	// Verify execution result
	if (is_verification_mode)
		iir.Verify();

	// Summarize
	iir.Summarize();
}
