#include <cstring>

#include "IirFilter.h"

void DumpHelpInfo()
{
	static const char help_info[] = 
		"IIR Filter benchmark\n\n" 
		"Syntax: IIR [options]\n\n" 
		"Options: \n" 
		"-h --help: Show this page.\n" 
		"-l --length: Data length\n";
	printf("%s", help_info);
}

int main(int argc, char **argv) 
{
	uint32_t len = 1024;
	bool is_verification_mode = false;

	// Parse argument
	for (int i = 1; i < argc; i++)
	{
		if (strcmp("-h", argv[i]) == 0 ||
			strcmp("--help", argv[i]) == 0)
		{
			DumpHelpInfo();
			exit(0);
		}
		else if (strcmp("-l", argv[i]) == 0 || 
			strcmp("--length", argv[i]) == 0)
		{
			if (i == argc - 1) 
			{
				DumpHelpInfo();
				exit(0);
			}
			else 
			{
				try 
				{
					len = stoi(std::string(argv[i+1]));
				}
				catch (std::exception) 
				{
					printf("Invalid number for argument "
							"len\n");
					DumpHelpInfo();
					exit(0);
				}
				i++;
			}
		}
		else if (strcmp("-v", argv[i]) == 0 ||
			strcmp("--verify", argv[i]) == 0)
		{
			is_verification_mode = true;
		}
		else
		{
			printf("Unknown argument\n");
			DumpHelpInfo();
			exit(1);
		}
	}

	// Set helper
	HsaHelper helper = HsaHelper();
	helper.setVerificationMode(is_verification_mode);	

	IirFilter iir = IirFilter();
	iir.setHelper(&helper);
	iir.setDataLength(len);
	iir.Init();
	iir.Run();
	if (is_verification_mode)
		iir.Verify();
}
