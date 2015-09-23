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

#include "src/common/runtime_helper/hsa_runtime_helper/hsa_agent.h"

HsaAgent::HsaAgent(hsa_agent_t agent) :
  agent_(agent) {
}

const std::string HsaAgent::GetNameOrDie() {
  char name[64] = {0};
  hsa_agent_get_info(agent_, HSA_AGENT_INFO_NAME, name);
  return std::string(name);
}

AqlQueue *HsaAgent::CreateQueueOrDie() {
  hsa_queue_t *queue;
  hsa_queue_create(agent_, 100, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, 
      UINT32_MAX, UINT32_MAX, &queue);

  auto aql_queue_unique = std::unique_ptr<AqlQueue>(new AqlQueue(queue));
  AqlQueue *aql_queue_ptr = aql_queue_unique.get();
  queues_.push_back(std::move(aql_queue_unique));
  return aql_queue_ptr;
}

hsa_isa_t HsaAgent::GetIsa() {
  hsa_isa_t isa;
  hsa_agent_get_info(agent_, HSA_AGENT_INFO_ISA, &isa);
  return isa;
}
