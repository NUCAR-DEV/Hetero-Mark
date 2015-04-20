#ifndef COMMON_OPTIONPARSER_H
#define COMMON_OPTIONPARSER_H

template <typename T>
class Argument
{
protected: 

	// Name of the argument
	std::string name;

	// Short format
	std::string short_format;

	// Long format
	std::string long_format;

	// Value
	T value;

public: 

	T getValue() const { return value; }

}

/**
 * Option parser is an 
 */
class OptionParser 
{
protected: 


public:
}

#endif
