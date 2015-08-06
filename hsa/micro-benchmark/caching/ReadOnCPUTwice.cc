/*
 * hetero-mark
 *
 * copyright (c) 2015 northeastern university
 * all rights reserved.
 *
 * developed by:
 *   northeastern university computer architecture research (nucar) group
 *   northeastern university
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * author: yifan sun (yifansun@coe.neu.edu)
 *
 * permission is hereby granted, free of charge, to any person obtaining a 
 * copy of this software and associated documentation files (the "software"), 
 * to deal with the software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the software, and to permit persons to whom the 
 * software is furnished to do so, subject to the following conditions:
 * 
 *   redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimers.
 *
 *   redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimers in the 
 *   documentation and/or other materials provided with the distribution.
 *
 *   neither the names of nucar, northeastern university, nor the names of 
 *   its contributors may be used to endorse or promote products derived 
 *   from this software without specific prior written permission.
 *
 * the software is provided "as is", without warranty of any kind, express or 
 * implied, including but not limited to the warranties of merchantability, 
 * fitness for a particular purpose and noninfringement. in no event shall the 
 * contributors or copyright holders be liable for any claim, damages or other 
 * liability, whether in an action of contract, tort or otherwise, arising 
 * from, out of or in connection with the software or the use or other 
 * dealings with the software.
 */

#include "hsa/micro-benchmark/caching/ReadOnCPUTwice.h"

void ReadOnCPUTwice::initialize() {
  input = new uint32_t[size];
}

void ReadOnCPUTwice::run() {
  for (uint64_t i = 0; i < size; i++) {
      output = input[i];
      output = input[i];
  }
}

void ReadOnCPUTwice::verify() {
}

void ReadOnCPUTwice::summarize() {
}

void ReadOnCPUTwice::cleanUp() {
  delete[] input;
}

