#include <cstring>

#include "../common/OptionParser.h"

#include "HMM.h"

uint32_t N = 16;
bool is_verification_mode = false;

void ParseArgument(int argc, char **argv)
{
	OptionParser parser = OptionParser(argv[0], 
			"The IIR filter benchmark");

	// Set length argument
	parser.AddArgument(
			"N", 
			"-n", "--num_hidden_states",
			"The number of hidden states", 
			"16",
			typeid(unsigned int)
			);

	// Set verification mode argument
	parser.AddArgument(
			"Verification Mode", 
			"-v", "--verify",
			"Enter verification mode.\n"
			"Output will be elliminated if in benchmark mode.", 
			"false",
			typeid(bool)
			);	

	// Parse argument
	parser.Parse(argc, argv);

	// Get length
	N = parser.getValue<uint32_t>("N");

	// Get verification mode
	is_verification_mode = parser.getValue<bool>("Verification Mode");
}


int main(int argc, char **argv) 
{
	// Parse argument
	ParseArgument(argc, argv);

	// Setup runtime helper
	HsaHelper helper = HsaHelper();
	helper.setVerificationMode(is_verification_mode);

	// Init benchmark
	HMM hmm = HMM();
	hmm.setHelper(&helper);
	hmm.setNumHiddenState(N);
	hmm.Init();

	// Run benchmark
	hmm.Run();

	// Verify execution result
	if (is_verification_mode)
		hmm.Verify();

	// Summarize
	hmm.Summarize();
}
