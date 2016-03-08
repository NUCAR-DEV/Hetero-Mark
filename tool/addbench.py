#!/usr/bin/env python

import argparse
import os
import sys


class Benchmark():

    root_cmakelists = '/template/root_cmakelists.tpl'
    root_benchmark_h = '/template/root_benchmark_h.tpl'
    root_benchmark_cc = '/template/root_benchmark_cc.tpl'
    root_commandline_option_h = '/template/root_commandline_option_h.tpl'
    root_commandline_option_cc = '/template/root_commandline_option_cc.tpl'

    cl12_cmakelists = '/template/cl12/cl12_cmakelists.tpl'
    cl12_benchmark_h = '/template/cl12/cl12_benchmark_h.tpl'
    cl12_benchmark_cc = '/template/cl12/cl12_benchmark_cc.tpl'
    cl12_kernel_cl = '/template/cl12/cl12_kernel_cl.tpl'
    cl12_main_cc = '/template/cl12/cl12_main_cc.tpl'

    cl20_cmakelists = '/template/cl20/cl20_cmakelists.tpl'
    cl20_benchmark_h = '/template/cl20/cl20_benchmark_h.tpl'
    cl20_benchmark_cc = '/template/cl20/cl20_benchmark_cc.tpl'
    cl20_kernel_cl = '/template/cl20/cl20_kernel_cl.tpl'
    cl20_main_cc = '/template/cl20/cl20_main_cc.tpl'

    hsa_cmakelists = '/template/hsa/hsa_cmakelists.tpl'
    hsa_benchmark_h = '/template/hsa/hsa_benchmark_h.tpl'
    hsa_benchmark_cc = '/template/hsa/hsa_benchmark_cc.tpl'
    hsa_kernel_cl = '/template/hsa/hsa_kernel_cl.tpl'
    hsa_main_cc = '/template/hsa/hsa_main_cc.tpl'

    def __init__(self, name):

        self.name_ = str(name)
        self.exec_path_ = os.path.dirname(sys.argv[0])
        self.dump_root_ = self.exec_path_ + "/../src/" + self.name_ + "/"

        if not os.path.isdir(self.dump_root_):
            os.makedirs(self.dump_root_)
            os.makedirs(self.dump_root_ + '/cl12')
            os.makedirs(self.dump_root_ + '/cl20')
            os.makedirs(self.dump_root_ + '/hsa')

    def getFileString(self, path):
        with open(path, 'r') as file:
            return file.read()

    def updateBenchmarkName(self, tpl_path):
        output = self.getFileString(tpl_path)
        output = output.replace("BENCHNAMEUPPER", self.name_.upper())
        output = output.replace("BENCHNAMELOWER", self.name_.lower())
        output = output.replace("BENCHNAMECAP", self.name_.capitalize())
        return output

    def getRootCMakeLists(self):
        tpl_path = self.exec_path_ + self.root_cmakelists
        return self.updateBenchmarkName(tpl_path)

    def getRootCommandlineOptionH(self):
        tpl_path = self.exec_path_ + self.root_commandline_option_h
        return self.updateBenchmarkName(tpl_path)

    def getRootCommandlineOptionCC(self):
        tpl_path = self.exec_path_ + self.root_commandline_option_cc
        return self.updateBenchmarkName(tpl_path)

    def getRootBenchmarkH(self):
        tpl_path = self.exec_path_ + self.root_benchmark_h
        return self.updateBenchmarkName(tpl_path)

    def getRootBenchmarkCC(self):
        tpl_path = self.exec_path_ + self.root_benchmark_cc
        return self.updateBenchmarkName(tpl_path)

    def dumpRootCmakeLists(self):
        output = open(self.dump_root_ + "CMakeLists.txt", "w")
        output.write(self.getRootCMakeLists())
        output.close()

    def dumpRootBenchmarkH(self):
        output = open(self.dump_root_ + self.name_ + "_benchmark.h", "w")
        output.write(self.getRootBenchmarkH())
        output.close()

    def dumpRootBenchmarkCC(self):
        output = open(self.dump_root_ + self.name_ + "_benchmark.cc", "w")
        output.write(self.getRootBenchmarkCC())
        output.close()

    def dumpRootCommandlineOptionH(self):
        output = open(self.dump_root_ + self.name_ +
                      "_command_line_options.h", "w")
        output.write(self.getRootCommandlineOptionH())
        output.close()

    def dumpRootCommandlineOptionCC(self):
        output = open(self.dump_root_ + self.name_ +
                      "_command_line_options.cc", "w")
        output.write(self.getRootCommandlineOptionCC())
        output.close()

    def dumpRootAll(self):
        self.dumpRootCmakeLists()
        self.dumpRootBenchmarkH()
        self.dumpRootBenchmarkCC()
        self.dumpRootCommandlineOptionH()
        self.dumpRootCommandlineOptionCC()

    def getCL12CMakeLists(self):
        tpl_path = self.exec_path_ + self.cl12_cmakelists
        return self.updateBenchmarkName(tpl_path)

    def getCL12BenchmarkH(self):
        tpl_path = self.exec_path_ + self.cl12_benchmark_h
        return self.updateBenchmarkName(tpl_path)

    def getCL12BenchmarkCC(self):
        tpl_path = self.exec_path_ + self.cl12_benchmark_cc
        return self.updateBenchmarkName(tpl_path)

    def getCL12Kernel(self):
        tpl_path = self.exec_path_ + self.cl12_kernel_cl
        return self.updateBenchmarkName(tpl_path)

    def getCL12MainCC(self):
        tpl_path = self.exec_path_ + self.cl12_main_cc
        return self.updateBenchmarkName(tpl_path)

    def dumpCL12MakeLists(self):
        output = open(self.dump_root_ + "cl12/CMakeLists.txt", "w")
        output.write(self.getCL12CMakeLists())
        output.close()

    def dumpCL12BenchmarkH(self):
        output = open(self.dump_root_ + "cl12/" +
                      self.name_ + "_cl12_benchmark.h", "w")
        output.write(self.getCL12BenchmarkH())
        output.close()

    def dumpCL12BenchmarkCC(self):
        output = open(self.dump_root_ + "cl12/" +
                      self.name_ + "_cl12_benchmark.cc", "w")
        output.write(self.getCL12BenchmarkCC())
        output.close()

    def dumpCL12MainCC(self):
        output = open(self.dump_root_ + "cl12/main.cc", "w")
        output.write(self.getCL12MainCC())
        output.close()

    def dumpCL12Kernel(self):
        output = open(self.dump_root_ + "cl12/kernel.cl", "w")
        output.write(self.getCL12Kernel())
        output.close()

    def dumpCL12All(self):
        self.dumpCL12MakeLists()
        self.dumpCL12BenchmarkH()
        self.dumpCL12BenchmarkCC()
        self.dumpCL12MainCC()
        self.dumpCL12Kernel()

    def getCL20CMakeLists(self):
        tpl_path = self.exec_path_ + self.cl20_cmakelists
        return self.updateBenchmarkName(tpl_path)

    def getCL20BenchmarkH(self):
        tpl_path = self.exec_path_ + self.cl20_benchmark_h
        return self.updateBenchmarkName(tpl_path)

    def getCL20BenchmarkCC(self):
        tpl_path = self.exec_path_ + self.cl20_benchmark_cc
        return self.updateBenchmarkName(tpl_path)

    def getCL20Kernel(self):
        tpl_path = self.exec_path_ + self.cl20_kernel_cl
        return self.updateBenchmarkName(tpl_path)

    def getCL20MainCC(self):
        tpl_path = self.exec_path_ + self.cl20_main_cc
        return self.updateBenchmarkName(tpl_path)

    def dumpCL20MakeLists(self):
        output = open(self.dump_root_ + "cl20/CMakeLists.txt", "w")
        output.write(self.getCL20CMakeLists())
        output.close()

    def dumpCL20BenchmarkH(self):
        output = open(self.dump_root_ + "cl20/" +
                      self.name_ + "_cl20_benchmark.h", "w")
        output.write(self.getCL20BenchmarkH())
        output.close()

    def dumpCL20BenchmarkCC(self):
        output = open(self.dump_root_ + "cl20/" +
                      self.name_ + "_cl20_benchmark.cc", "w")
        output.write(self.getCL20BenchmarkCC())
        output.close()

    def dumpCL20MainCC(self):
        output = open(self.dump_root_ + "cl20/main.cc", "w")
        output.write(self.getCL20MainCC())
        output.close()

    def dumpCL20Kernel(self):
        output = open(self.dump_root_ + "cl20/kernel.cl", "w")
        output.write(self.getCL20Kernel())
        output.close()

    def dumpCL20All(self):
        self.dumpCL20MakeLists()
        self.dumpCL20BenchmarkH()
        self.dumpCL20BenchmarkCC()
        self.dumpCL20MainCC()
        self.dumpCL20Kernel()

    def getHSACMakeLists(self):
        tpl_path = self.exec_path_ + self.hsa_cmakelists
        return self.updateBenchmarkName(tpl_path)

    def getHSABenchmarkH(self):
        tpl_path = self.exec_path_ + self.hsa_benchmark_h
        return self.updateBenchmarkName(tpl_path)

    def getHSABenchmarkCC(self):
        tpl_path = self.exec_path_ + self.hsa_benchmark_cc
        return self.updateBenchmarkName(tpl_path)

    def getHSAKernel(self):
        tpl_path = self.exec_path_ + self.hsa_kernel_cl
        return self.updateBenchmarkName(tpl_path)

    def getHSAMainCC(self):
        tpl_path = self.exec_path_ + self.hsa_main_cc
        return self.updateBenchmarkName(tpl_path)

    def dumpHSAMakeLists(self):
        output = open(self.dump_root_ + "hsa/CMakeLists.txt", "w")
        output.write(self.getHSACMakeLists())
        output.close()

    def dumpHSABenchmarkH(self):
        output = open(self.dump_root_ + "hsa/" +
                      self.name_ + "_hsa_benchmark.h", "w")
        output.write(self.getHSABenchmarkH())
        output.close()

    def dumpHSABenchmarkCC(self):
        output = open(self.dump_root_ + "hsa/" +
                      self.name_ + "_hsa_benchmark.cc", "w")
        output.write(self.getHSABenchmarkCC())
        output.close()

    def dumpHSAMainCC(self):
        output = open(self.dump_root_ + "hsa/main.cc", "w")
        output.write(self.getHSAMainCC())
        output.close()

    def dumpHSAKernel(self):
        output = open(self.dump_root_ + "hsa/kernel.cl", "w")
        output.write(self.getHSAKernel())
        output.close()

    def dumpHSAAll(self):
        self.dumpHSAMakeLists()
        self.dumpHSABenchmarkH()
        self.dumpHSABenchmarkCC()
        self.dumpHSAMainCC()
        self.dumpHSAKernel()


def main():
    # Arg parser
    parser = argparse.ArgumentParser(
        description='HeteroMark development helper tool')
    parser.add_argument('name', nargs=1,
                        help='Name of benchmark to be added to HeteroMark')
    args = parser.parse_args()

    bench = Benchmark(args.name[0])
    bench.dumpRootAll()
    bench.dumpCL12All()
    bench.dumpCL20All()
    bench.dumpHSAAll()


if __name__ == '__main__':
    main()
