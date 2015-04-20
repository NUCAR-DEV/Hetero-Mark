#ifndef COMMON_OPTIONPARSER_H
#define COMMON_OPTIONPARSER_H

#include <string>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <memory>
#include <cxxabi.h>

class Argument
{
protected: 

	// Name of the argument
	std::string name;

	// Short format
	std::string short_format;

	// Long format
	std::string long_format;

	// Description of the argument
	std::string description;

	// Default value
	std::string default_value;

	// Type of the input
	const std::type_info &type;

	// Human readable name of the type
	std::string type_name;

	// Value, store the original input string
	std::string value;

public: 

	/**
	 * Constructor 
	 */
	Argument(const char *name, 
			const char *short_format, const char *long_format, 
			const char *description, const char *default_value,
			const std::type_info &type);

	/**
	 * Get the value
	 */
	template<typename T>
	T getValue() const
	{
		// Requested type should be the same with the argument type
		if (typeid(T) != type)
		{
			printf("Trying to get value of argument %s in type %s, but "
				"should be %s", name.c_str(), 
				abi::__cxa_demangle(typeid(T).name(), 0, 0, NULL),
				type_name.c_str());
			exit(1);
		}

		// Return value according to the type
		if (type == typeid(bool))
		{
			if (value == "true")
				return true;
			else
				return false;
		}
		else if (type == typeid(int) || type == typeid(unsigned int))
		{
			int ret;
			try 
			{
				ret = stoi(value);
			}
			catch (std::exception e)
			{
				printf("Invalid argument value %s.\n", value.c_str());
				exit(1);
			}
			return ret;
		}
		else 
		{
			printf("Unspoorted argument type.\n");
			exit(1);
		}

	}


	/** 
	 * Dump help infomation
	 */
	void DumpHelp() const;

	/**
	 * Parse argument 
	 * 
	 * @param index 
	 * 	The index from where the argument should try to parse
	 *
	 * @return 
	 *	The number of arguments this parsing consumes
	 *
	 */
	int Parse(int argc, char **argv, int index);

};

/**
 * Option parser is a tool that helps define and parse the command line 
 * arguments.
 */
class OptionParser 
{
protected: 

	// The program name
	std::string program_name;

	// The description of the program
	std::string description;

	// A list of arguments, indexed by the name of the argument
	std::unordered_map<std::string, std::unique_ptr<Argument>> arguments;

public:

	/**
	 * Constructor
	 */
	OptionParser(const char *program_name, const char *description) :
		program_name(program_name),
		description(description)
	{
		// Add an argument for help
		AddArgument(
			"Help", 
			"-h", "--help", 
			"Print help page", 
			"false", 
			typeid(bool));

	};

	/**
	 * Print help information
	 */
	void DumpHelp() const;

	/**
	 * Add an argument
	 */
	void AddArgument(const char *name, 
			const char *short_format, const char *long_format, 
			const char *description,
			const char *defalue_value,
			const std::type_info &type);

	/**
	 * Parse
	 */ 
	void Parse(int argc, char **argv);

	/**
	 * Get value
	 */
	template <typename T>
	T getValue (const char *name) const
	{
		auto it = arguments.find(name);
		
		// Check if such name exist
		if (it == arguments.end())
		{
			printf("Argument %s is not defined.\n", name);
			DumpHelp();
			exit(1);
		}

		// Get value
		return it->second->getValue<T>();
	}

};

#endif
