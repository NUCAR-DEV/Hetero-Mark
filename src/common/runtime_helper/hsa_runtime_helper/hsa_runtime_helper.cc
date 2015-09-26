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

#include "src/common/runtime_helper/hsa_runtime_helper/hsa_runtime_helper.h"

#include <cstring>
#include <iostream>

HsaRuntimeHelper::HsaRuntimeHelper(HsaErrorChecker *error_checker) :
  error_checker_(error_checker) {}

void HsaRuntimeHelper::InitializeOrDie() {
  status_ = hsa_init();
  error_checker_->SucceedOrDie("Initialize hsa runtime environment", status_);
}

HsaAgent *HsaRuntimeHelper::FindGpuOrDie() {
  if (!gpu_.get()){
    hsa_agent_t agent;
    status_ = hsa_iterate_agents(HsaRuntimeHelper::GetGpuIterateCallback, 
        &agent);
    if (status_ == HSA_STATUS_INFO_BREAK) { status_ = HSA_STATUS_SUCCESS; }
    error_checker_->SucceedOrDie("Find GPU agent", status_);
    gpu_.reset(new HsaAgent(agent, error_checker_));
  }

  return gpu_.get();
}

hsa_status_t HsaRuntimeHelper::GetGpuIterateCallback(
    hsa_agent_t agent, void *data) {
  hsa_status_t status;
  hsa_device_type_t device_type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
    hsa_agent_t* ret = (hsa_agent_t*)data;
    *ret = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

HsaExecutable *HsaRuntimeHelper::CreateProgramFromSourceOrDie(
    const char *filename, HsaAgent *agent) {
  // Load module from hard drive
  hsa_ext_module_t module;
  LoadBrigModuleFromFile(filename, &module);

  // Create a program
  hsa_ext_program_t program = CreateProgram();

  // Add the module to the program
  AddModuleToProgram(program, module);

  // Finalize
  hsa_isa_t isa = agent->GetIsa();
  hsa_code_object_t code_object = FinalizeProgram(program, isa);

  // Destory the program
  DestroyProgram(program);

  // Create an empty executable 
  hsa_executable_t executable = CreateExecutable();

  // Load code object into the executable
  ExecutableLoadCodeObject(executable, *(hsa_agent_t *)agent->GetNative(), 
      code_object);

  // Freeze the executable
  FreezeExecutable(executable);

  // Create executable object
  auto executable_unique = std::unique_ptr<HsaExecutable>(
      new HsaExecutable(executable, error_checker_));
  HsaExecutable *executable_ptr = executable_unique.get();
  executables_.push_back(std::move(executable_unique));

  // Return
  return executable_ptr;
}

int HsaRuntimeHelper::LoadBrigModuleFromFile(
    const char *file_name, hsa_ext_module_t *module) {
  int rc = -1;
  FILE *fp = fopen(file_name, "rb");
  if (!fp) {
    std::cerr << "Faile to open file " << file_name << ".\n";
    exit(1);
  }
  rc = fseek(fp, 0, SEEK_END);
  size_t file_size = (size_t) (ftell(fp) * sizeof(char));
  rc = fseek(fp, 0, SEEK_SET);
  char* buf = (char*) malloc(file_size);
  memset(buf,0,file_size);
  size_t read_size = fread(buf,sizeof(char),file_size,fp);
  if(read_size != file_size) {
    free(buf);
  } else {
    rc = 0;
    *module = (hsa_ext_module_t) buf;
  }
  fclose(fp);
  return rc;
}

hsa_ext_program_t HsaRuntimeHelper::CreateProgram() {
  hsa_ext_program_t program;
  memset(&program, 0, sizeof(hsa_ext_program_t));
  status_ = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, 
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &program);
  error_checker_->SucceedOrDie("Create the program", status_);
  return program;
}

void HsaRuntimeHelper::AddModuleToProgram(hsa_ext_program_t program, 
      hsa_ext_module_t module) {
  status_ = hsa_ext_program_add_module(program, module);
  error_checker_->SucceedOrDie("Adding the brig module to the program", 
      status_);
}

hsa_code_object_t HsaRuntimeHelper::FinalizeProgram(
    hsa_ext_program_t program, hsa_isa_t isa) {

  // Create empty control directives
  hsa_ext_control_directives_t control_directives;
  memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

  // Create code object
  hsa_code_object_t code_object;

  // Finalize
  status_ = hsa_ext_program_finalize(program, isa, 0, control_directives, "", 
      HSA_CODE_OBJECT_TYPE_PROGRAM, &code_object);
  error_checker_->SucceedOrDie("Finalizing the program", status_);

  // Return code object
  return code_object;
}

void HsaRuntimeHelper::DestroyProgram(hsa_ext_program_t program) {
  status_ = hsa_ext_program_destroy(program);
  error_checker_->SucceedOrDie("Destroying the program", status_);
}

hsa_executable_t HsaRuntimeHelper::CreateExecutable() {
  hsa_executable_t executable;
  status_ = hsa_executable_create(HSA_PROFILE_FULL, 
      HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
  error_checker_->SucceedOrDie("Create the executable", status_);
  return executable;
}

void HsaRuntimeHelper::ExecutableLoadCodeObject(hsa_executable_t executable,
    hsa_agent_t agent, hsa_code_object_t code_object) {
  status_ = hsa_executable_load_code_object(executable, agent, 
      code_object, "");
  error_checker_->SucceedOrDie("Loading the code object", status_);
}

void HsaRuntimeHelper::FreezeExecutable(hsa_executable_t executable) {
  status_ = hsa_executable_freeze(executable, "");
  error_checker_->SucceedOrDie("Freeze the executable", status_);
}

HsaSignal *HsaRuntimeHelper::CreateSignal(int64_t initial_value) {
  hsa_signal_t signal;
  status_ = hsa_signal_create(initial_value, 0, NULL, &signal);
  error_checker_->SucceedOrDie("Creating HSA signal.", status_);

  auto signal_unique = std::unique_ptr<HsaSignal>(new HsaSignal(signal));
  HsaSignal *signal_ptr = signal_unique.get();
  signals_.push_back(std::move(signal_unique));
  return signal_ptr;
}
