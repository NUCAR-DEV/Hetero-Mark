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
 * Author: Yifan Sun (yifansun@coe.neu.edu)
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

#ifndef SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_AGENT_H_
#define SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_AGENT_H_

#include <vector>
#include <memory>
#include <hsa.h>

#include "src/common/runtime_helper/agent.h"
#include "src/common/runtime_helper/hsa_runtime_helper/aql_queue.h"

class HsaAgent : public Agent {
 public:
  HsaAgent(hsa_agent_t agent_);
  virtual ~HsaAgent() {}
  const std::string GetNameOrDie() override;
  AqlQueue *CreateQueueOrDie() override;
  hsa_agent_t *GetNative() { return &agent_; }
  hsa_isa_t GetIsa();

 private:
  hsa_agent_t agent_;
  std::vector<std::unique_ptr<AqlQueue>> queues_;
};

#endif  // SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_AGENT_H_
