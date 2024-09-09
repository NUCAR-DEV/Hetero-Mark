#!/usr/bin/env python3

"""This file automatically runs all the benchmarks of Hetero-Mark
"""
from __future__ import print_function

import os
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
from benchmark import BEBenchmark
from benchmark import GABenchmark


def main():
    """main function"""
    args = parse_args()
    if not args.skip_build:
        compile_benchmark(args)

    benchmarks = []
    setup_benchmarks(benchmarks, args)
    run(benchmarks, args)


def parse_args():
    """parse user input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=str, default="./build", help="""
            Directory in which to build Hetero-Mark in.
    """)
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
            Set "-DHMARK_BUILD_CUDA=On" to enable CUDA compilation.
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

    return args


def compile_benchmark(args):
    compile_log_filename = "compile_log.txt"
    compile_log = open(compile_log_filename, "w")

    print("Compiling benchmark into", args.build_dir)
    if not os.path.exists(args.build_dir):
        os.makedirs(args.build_dir)

    if args.fresh_build:
        subprocess.call(['rm', '-rf', args.build_dir])
        subprocess.call(['mkdir', args.build_dir])

        env = os.environ.copy()
        env['CXX'] = args.cxx
        cmake_command = 'cmake '
        if args.cmake_flag:
            cmake_command += str(args.cmake_flag)
        p = subprocess.Popen(cmake_command + ' ' + os.getcwd(),
                             cwd=args.build_dir, env=env, shell=True,
                             stdout=compile_log, stderr=compile_log)
        p.wait()
        if p.returncode != 0:
            print(bcolors.FAIL + "Compile failed, see",
                  compile_log_filename, "for detailed information", bcolors.ENDC)
            exit(-1)

    p = subprocess.Popen('make -j VERBOSE=1',
                         cwd=args.build_dir, shell=True,
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
    benchmarks.append(BEBenchmark(args))
    benchmarks.append(GABenchmark(args))
    benchmarks.append(EPBenchmark(args))


def run(benchmarks, args):
    """ Run all benchmarks """
    for benchmark in benchmarks:

        if args.benchmark is not None and args.benchmark != benchmark.benchmark_name:
            continue

        benchmark.run()


if __name__ == "__main__":
    main()
