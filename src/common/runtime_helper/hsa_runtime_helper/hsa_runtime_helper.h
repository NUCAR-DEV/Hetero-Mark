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

#ifndef SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_RUNTIME_HELPER_H_
#define SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_RUNTIME_HELPER_H_

#include <memory>
#include <vector>
#include <hsa.h>
#include <hsa_ext_finalize.h>

#include "src/common/runtime_helper/runtime_helper.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_agent.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_error_checker.h"
#include "src/common/runtime_helper/hsa_runtime_helper/hsa_executable.h"

class HsaRuntimeHelper : public RuntimeHelper {
 public:
  HsaRuntimeHelper(HsaErrorChecker *error_checker);
  virtual ~HsaRuntimeHelper() {}

  void InitializeOrDie() override;
  HsaAgent *FindGpuOrDie() override;
  HsaExecutable *CreateProgramFromSourceOrDie(const char *filename,
      HsaAgent *agent);
 
 private:
  hsa_status_t status_;
  HsaErrorChecker *error_checker_;
  std::unique_ptr<HsaAgent> gpu_;
  std::vector<std::unique_ptr<HsaExecutable>> executables_;

  static hsa_status_t GetGpuIterateCallback(hsa_agent_t agent, void *data);

  int LoadBrigModuleFromFile(const char *filename, hsa_ext_module_t *module);
  hsa_ext_program_t CreateProgram();
  void AddModuleToProgram(
      hsa_ext_program_t program, hsa_ext_module_t module);
  hsa_code_object_t FinalizeProgram(hsa_ext_program_t program, hsa_isa_t isa);
  void DestroyProgram(hsa_ext_program_t program);
  hsa_executable_t CreateExecutable();
  void ExecutableLoadCodeObject(
      hsa_executable_t executable, hsa_agent_t agent, 
      hsa_code_object_t code_object);
  void FreezeExecutable(hsa_executable_t executable);

};

#endif  // SRC_COMMON_RUNTIME_HELPER_HSA_RUNTIME_HELPER_HSA_RUNTIME_HELPER_H_

