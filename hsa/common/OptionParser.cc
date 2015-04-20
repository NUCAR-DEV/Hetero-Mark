#include <map>
#include <cstring>
#include <exception>

#include "OptionParser.h"

Argument::Argument(const char *name, 
		const char *short_format, const char *long_format, 
		const char *description,
		const char *default_value,
		const std::type_info &type) :
	type(type)
{
	this->name = std::string(name);	
	this->short_format = std::string(short_format);
	this->long_format = std::string(long_format);
	this->description = std::string(description);
	this->default_value = default_value;
	this->value = default_value;
	this->type_name = abi::__cxa_demangle(type.name(), 0, 0, NULL);

}


void Argument::DumpHelp() const
{
	printf("\t%s: %s %s : %s (default = %s)\n\n\t\t%s \n\n", 
			name.c_str(), short_format.c_str(), 
			long_format.c_str(), type_name.c_str(), 
			default_value.c_str(),
			description.c_str());
}


int Argument::Parse(int argc, char **argv, int index)
{
	// Compare the argument
	if (strcmp(short_format.c_str(), argv[index]) != 0 &&
		strcmp(long_format.c_str(), argv[index]) != 0)
	{
		return 0;
	}

	// The format matches
	if (type == typeid(bool))
	{
		value = std::string("true");
		return 1;
	}
	else
	{
		if (index == argc - 1)
		{
			return -1;
		}
		value = argv[index + 1];
		return 2;
	}

}


void OptionParser::AddArgument(const char *name, 
		const char *short_format, const char *long_format, 
		const char *description, const char *default_value,
		const std::type_info &type)
{
	std::unique_ptr<Argument> argument = 
		std::unique_ptr<Argument>(new Argument(name, short_format, 
					long_format, description, 
					default_value, type));
	arguments.insert(std::make_pair<std::string, std::unique_ptr<Argument>>
			(name, std::move(argument)));
}


void OptionParser::DumpHelp() const
{
	// Print information about the program
	printf("\n%s [OPTIONS]: \n\n\t%s \n\n\nOPTIONS: \n\n",
			program_name.c_str(), description.c_str());

	// Print the information about the arguments
	for (auto it = arguments.begin(); it != arguments.end(); it++)
	{
		it->second->DumpHelp();
	}
}


void OptionParser::Parse(int argc, char **argv)
{
	int i = 1;
	while (i < argc)
	{
		int ret = 0;
		if (strcmp(argv[i], "-h") == 0 || 
			strcmp(argv[i], "--help") ==0)
		{
			DumpHelp();
			exit(0);
		}

		// Go throught each argument to find a match
		for (auto it = arguments.begin(); it != arguments.end(); it++)
		{
			ret = it->second->Parse(argc, argv, i);

			// If return value is below 0, error
			if (ret < 0)
			{
				DumpHelp();
				exit(1);
			}
			// If return value is greater than 0, then parsed
			else if (ret > 0)
			{
				i += ret;
				break;
			}
		}

		// Argument is not recognized, error
		if (ret = 0)
		{
			DumpHelp();
			exit(1);
		}
	}
}
