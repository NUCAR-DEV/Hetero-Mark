/**
 * CommandLineOption is an facade of the command line parser system. It 
 * provides all the interfaces required for users of the command line parser
 * system. 
 */
class CommandLineOption {
 public:

  // Add an argument of type 32 bit integer
  void addArgumentInt32();

  // Add an argument of type 64 bit integer
  void addArgumentInt64();

  // Add an argument of type double 
  void addArgumentDouble();

  // Add an argument of type float
  void addArgumentFloat();

  // Add an argument of type string
  void addArgumentString();

  // Add an argument of type enum
  void addArgumentEnum();

  // Add an arugment of type boolean
  void addArgumentBool();

  // Parse
  void Parse();

  // Get value of an argument
  template <typename T> 
  T getValue(const char *argumentName);
}
