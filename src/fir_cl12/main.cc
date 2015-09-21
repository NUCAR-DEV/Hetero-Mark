/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Carter McCardwell (cmccardw@coe.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a 
 * copy of this software and associated documentation files (the "Software"), 
 * to deal with the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 * 
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimers in the 
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of 
 *   its contributors may be used to endorse or promote products derived 
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */


#include "FIR_CL.h"
#include "../common/benchmark/benchmark_runner.h"
#include "../common/time_measurement/time_measurement.h"
#include "../common/time_measurement/time_measurement_impl.h"
#include "../common/command_line_option/command_line_option.h"

#include <cstdlib>
#include <string>

int main(int argc, char const *argv[])
{
    // Setup command line option
    CommandLineOption command_line_option(
      "====== Hetero-Mark FIR Benchmarks (OpenCL 1.2) ======",
      "This benchmarks runs the FIR-Filter Algorithm.");
    command_line_option.AddArgument("Help", "bool", "false",
        "-h", "--help", "Dump help information");
    command_line_option.AddArgument("NumBlocks", "int", "100",
        "-b", "--blocks",
        "Number of test blocks");
    command_line_option.AddArgument("NumData", "int", "1000",
        "-d", "--data",
        "Number of data samples");

    command_line_option.Parse(argc, argv);
    if (command_line_option.GetArgumentValue("Help")->AsBool()) {
      command_line_option.Help();
      return 0;
    }

    std::unique_ptr<FIR> fir(new FIR());
    fir->SetInitialParameters(command_line_option.GetArgumentValue("NumData")->AsInt32(),\
     command_line_option.GetArgumentValue("NumBlocks")->AsInt32());

    std::unique_ptr<TimeMeasurement> timer(new TimeMeasurementImpl());
    BenchmarkRunner runner(fir.get(), timer.get());
    runner.Run();
    runner.Summarize();    
    
    return 0;
}
