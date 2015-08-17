#ifndef CL_FILE_H
#define CL_FILE_H

#include <memory>
#include <fstream>
#include <stdio.h>

namespace clHelper
{

class clFile
{

private:
        // Instance of the singleton
        static std::unique_ptr<clFile> instance;

        // Private constructor for singleton
        clFile(): sourceCode("") { }

        // Disable copy constructor
        clFile(const clFile&) = delete;

        // Disable operator=
        clFile& operator=(const clFile&) = delete;

        // source code of the CL program
        std::string sourceCode;

public:

        // Get singleton
        static clFile *getInstance();

        ~clFile() {};

        // Getters
        const std::string& getSource() const { return sourceCode; }

        const char *getSourceChar() const { return sourceCode.c_str(); }

        // Read file
        bool open(const char *fileName);

};

// Singleton instance
std::unique_ptr<clFile> clFile::instance;

clFile *clFile::getInstance()
{
        // Instance already exists
        if (instance.get())
                return instance.get();
        
        // Create instance
        instance.reset(new clFile());
        return instance.get();
}


bool clFile::open(const char *fileName)
{
        size_t size;
        char*  str;

        // Open file stream
        std::fstream f(fileName, (std::fstream::in | std::fstream::binary));

        // Check if we have opened file stream
        if (f.is_open())
        {
                size_t  sizeFile;

                // Find the stream size
                f.seekg(0, std::fstream::end);
                size = sizeFile = (size_t)f.tellg();
                f.seekg(0, std::fstream::beg);
                str = new char[size + 1];
                if (!str)
                {
                    f.close();
                    return false;
                }

                // Read file
                f.read(str, sizeFile);
                f.close();
                str[size] = '\0';
                sourceCode  = str;
                delete[] str;

                return true;
        }

        return false;

}

} // namespace clHelper

#endif
