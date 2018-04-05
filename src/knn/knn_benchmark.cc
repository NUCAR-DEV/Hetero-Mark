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

#include "src/knn/knn_benchmark.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "src/knn/knn_cpu_partitioner.h"

int KnnBenchmark::loadData(std::string file, std::vector<Record> *records,
                           std::vector<LatLong> *locations) {
  FILE *flist, *fp;
  int i = 0;
  char dbname[64];
  int recNum = 0;

  /**Main processing **/
  const char *filename = file.c_str();
  flist = fopen(filename, "r");
  while (!feof(flist)) {
    /**
    * Read in all records of length REC_LENGTH
    * If this is the last file in the filelist, then done
    * else open next file to be read next iteration
    */
    if (fscanf(flist, "%s\n", dbname) != 1) {
      fprintf(stderr, "error reading filelist\n");
      exit(0);
    }
    printf("dbname is %s \n", dbname);
    fp = fopen(dbname, "r");
    if (!fp) {
      printf("error opening a db\n");
      exit(1);
    }
    // read each record
    while (!feof(fp)) {
      Record record;
      LatLong latLong;
      fgets(record.recString, 49, fp);
      fgetc(fp);  // newline
      if (feof(fp)) break;

      // parse for lat and long
      char substr[6];

      for (i = 0; i < 5; i++) substr[i] = *(record.recString + i + 28);
      substr[5] = '\0';
      latLong.lat = atof(substr);

      for (i = 0; i < 5; i++) substr[i] = *(record.recString + i + 33);
      substr[5] = '\0';
      latLong.lng = atof(substr);

      locations->push_back(latLong);
      records->push_back(record);
      recNum++;
    }
    fclose(fp);
  }
  fclose(flist);
  return recNum;
}

void KnnBenchmark::findLowest(std::vector<Record> *records, float *distances,
                              int numRecords, int topN) {
  int i, j;
  float val;
  int minLoc;
  float *temp_distances = new float[num_records_];
  for (i = 0; i < numRecords; i++) {
    temp_distances[i] = distances[i];
  }
  Record *tempRec;
  float tempDist;

  for (i = 0; i < topN; i++) {
    minLoc = i;
    for (j = i; j < numRecords; j++) {
      val = temp_distances[j];
      if (val < temp_distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &(*records)[i];
    (*records)[i] = (*records)[minLoc];
    (*records)[minLoc] = *tempRec;

    tempDist = temp_distances[i];
    temp_distances[i] = temp_distances[minLoc];
    temp_distances[minLoc] = tempDist;

    // add distance to the min we just found
    (*records)[i].distance = temp_distances[i];
  }
}

void KnnBenchmark::Initialize() {
  num_records_ = loadData(filename_, &records_, &locations_);
  if (k_value_ > num_records_) {
    k_value_ = num_records_;
  }
}

void KnnBenchmark::KnnCPU(LatLong *latLong, float *d_distances, int num_records,
                          int num_gpu_records, float lat, float lng,
                          std::atomic_int *cpu_worklist,
                          std::atomic_int *gpu_worklist) {
  CpuPartitioner p = cpu_partitioner_create(num_records, cpu_worklist);

  for (int tid = cpu_initializer(&p); cpu_more(&p); tid = cpu_increment(&p)) {
    d_distances[tid] = static_cast<float>(
        sqrt((lat - latLong[tid].lat) * (lat - latLong[tid].lat) +
             (lng - latLong[tid].lng) * (lng - latLong[tid].lng)));
  }

  CpuPartitioner thieves =
      cpu_partitioner_create(num_gpu_records, gpu_worklist);

  for (int tid = cpu_initializer(&thieves); cpu_more(&thieves);
       tid = cpu_increment(&thieves)) {
    d_distances[tid] = static_cast<float>(
        sqrt((lat - latLong[tid].lat) * (lat - latLong[tid].lat) +
             (lng - latLong[tid].lng) * (lng - latLong[tid].lng)));
  }
}

void KnnBenchmark::Verify() {
  bool has_error = false;
  float *cpu_output = new float[num_records_];

  for (int i = 0; i < num_records_; i++) {
    cpu_output[i] =
        static_cast<float>(sqrt((latitude_ - locations_.at(i).lat) *
                                    (latitude_ - locations_.at(i).lat) +
                                (longitude_ - locations_.at(i).lng) *
                                    (longitude_ - locations_.at(i).lng)));
    if (std::abs(cpu_output[i] - h_distances_[i]) > 1e-2) {
      has_error = true;
      printf("At position %d , expected %f but is %f \n", i, cpu_output[i],
             h_distances_[i]);
      exit(1);
    }
  }

  if (!has_error) {
    printf("All distances correctly matched %d \n", num_records_);
  }
}
void KnnBenchmark::Summarize() {}

void KnnBenchmark::Cleanup() {}
