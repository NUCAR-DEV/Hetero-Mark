#!/usr/bin/env python

"""This file automatically runs all the benchmarks of Heter-Mark
"""
from __future__ import print_function

import os
import re
import sys
import subprocess
import argparse
from bcolors import bcolors
from benchmark import FirBenchmark
from benchmark import AesBenchmark
from benchmark import HistBenchmark
from benchmark import PRBenchmark
from benchmark import KMeansBenchmark
from benchmark import BSBenchmark
from benchmark import EPBenchmark

build_folder = os.getcwd() + '/build-auto-run/'
# benchmarks = [
#     ('be', 'hc', ['-i', os.getcwd() + '/data/be/0.mp4']),
#     ('be', 'hc', ['-i', os.getcwd() + '/data/be/0.mp4', '--collaborative']),
#     ('be', 'hc', ['-i', os.getcwd() + '/data/be/1.mp4']),
#     ('be', 'hc', ['-i', os.getcwd() + '/data/be/1.mp4', '--collaborative']),
#     ('be', 'cuda', ['-i', os.getcwd() + '/data/be/0.mp4']),
#     ('be', 'cuda', ['-i', os.getcwd() + '/data/be/0.mp4', '--collaborative']),
#     ('be', 'cuda', ['-i', os.getcwd() + '/data/be/1.mp4']),
#     ('be', 'cuda', ['-i', os.getcwd() + '/data/be/1.mp4', '--collaborative']),
#     ('be', 'hip', ['-i', os.getcwd() + '/data/be/0.mp4']),
#     ('be', 'hip', ['-i', os.getcwd() + '/data/be/0.mp4', '--collaborative']),
#     ('be', 'hip', ['-i', os.getcwd() + '/data/be/1.mp4']),
#     ('be', 'hip', ['-i', os.getcwd() + '/data/be/1.mp4', '--collaborative']),

#     ('ga', 'hc', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'hc', ['-i', os.getcwd() +
#                   '/data/gene_alignment/medium.data', '--collaborative']),
#     ('ga', 'cuda', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'cuda', ['-i', os.getcwd() +
#                     '/data/gene_alignment/medium.data', '--collaborative']),
#     ('ga', 'hip', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'hip', ['-i', os.getcwd() +
#                    '/data/gene_alignment/medium.data', '--collaborative']),


#     ('pr', 'cl12', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
#     ('pr', 'cl20', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
#     ('pr', 'hc', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
# ]

def main():
    """main function"""
    args = parse_args()
    if not args.skip_build:
        compile(args)

    benchmarks = []
    setup_benchmarks(benchmarks, args)
    run(benchmarks, args)


def parse_args():
    """parse user input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true",
                        help="""
            By default, the script performs an incremental build. 
            Setting this argument will skip the compilation process. This is 
            useful if you have the compiled with this script before.
            """)
    parser.add_argument("--fresh-build", action="store_true",
                        help="""
            Remove the temp build folder and build from scratch.
            """)
    parser.add_argument("--cmake-flag",
                        help="""
            Use this option to set the flags to pass to cmake. 
            Set "-DCOMPILE_CUDA=On" to enable CUDA compilation.
            """)
    parser.add_argument("--cxx", default="g++",
                        help="""
            The compiler to be used to compile the benchmark. 
            """)
    parser.add_argument("-i", "--ignore-error", action="store_true",
                            help="""
            Use this option to ignore errors in the compilation and 
            verification process.
            """)


    parser.add_argument("--skip-verification", action="store_true",
                        help="""
            Setting this argument will skip the CPU verification process.
            """)
    parser.add_argument("--full-verification", action="store_true",
                        help="""
            Perform a full verification on different input values. 
            """)
    parser.add_argument("-b", "--benchmark",
                        help="""
            Benchmark to run. By default, this script will run all the 
            benchmarks. Which this argument, you can specify a certain 
            benchmark to run.
            """)
    parser.add_argument("-r", "--repeat-time", default=5, type=int,
                        help="""
            The number of times to run a benchmark. Default is 5 times.
            """)
    args = parser.parse_args()

    args.build_folder = build_folder
    return args


def compile(args):
    compile_log_filename = "compile_log.txt"
    compile_log = open(compile_log_filename, "w")

    print("Compiling benchmark into", build_folder)

    if args.fresh_build:
        subprocess.call(['rm', '-rf', build_folder])
        subprocess.call(['mkdir', build_folder])

        env = os.environ.copy()
        env['CXX'] = args.cxx
        cmake_command = 'cmake '
        if args.cmake_flag:
            cmake_command += str(args.cmake_flag)
        p = subprocess.Popen(cmake_command + ' ' + os.getcwd(),
                             cwd=build_folder, env=env, shell=True,
                             stdout=compile_log, stderr=compile_log)
        p.wait()
        if p.returncode != 0:
            print(bcolors.FAIL + "Compile failed, see",
                  compile_log_filename, "for detailed information", bcolors.ENDC)
            exit(-1)

    p = subprocess.Popen('make -j VERBOSE=1',
                         cwd=build_folder, shell=True,
                         stdout=compile_log, stderr=compile_log)
    p.wait()
    if p.returncode != 0:
        print(bcolors.FAIL + "Compile failed, see", compile_log_filename,
              "for detailed information", bcolors.ENDC)
        if not args.ignore_error:
            exit(-1)
    else:
        print(bcolors.OKGREEN + "Compile completed." + bcolors.ENDC)


def setup_benchmarks(benchmarks, args):
    """List all the benchmarks"""
    benchmarks.append(FirBenchmark(args))
    benchmarks.append(AesBenchmark(args))
    benchmarks.append(HistBenchmark(args))
    benchmarks.append(PRBenchmark(args))
    benchmarks.append(KMeansBenchmark(args))
    benchmarks.append(BSBenchmark(args))
    benchmarks.append(EPBenchmark(args))


def run(benchmarks, args):
    """ Run all benchmarks """
    for benchmark in benchmarks:

        if args.benchmark != None and args.benchmark != benchmark.benchmark_name:
            continue

        benchmark.run()


if __name__ == "__main__":
    main()
