#!/usr/bin/env python

"""This file automatically runs all the benchmarks of Heter-Mark
"""
from __future__ import print_function

import os
import re
import sys
import subprocess
import argparse
from benchmark import FirBenchmark
from bcolors import bcolors

build_folder = os.getcwd() + '/build-auto-run/'
# benchmarks = [
#     ('aes', 'cl12', [
#         '-i', os.getcwd() + '/data/aes/small.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cl12', [
#         '-i', os.getcwd() + '/data/aes/medium.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cl12', [
#         '-i', os.getcwd() + '/data/aes/large.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cl20', [
#         '-i', os.getcwd() + '/data/aes/small.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cl20', [
#         '-i', os.getcwd() + '/data/aes/medium.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cl20', [
#         '-i', os.getcwd() + '/data/aes/large.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'hc', [
#         '-i', os.getcwd() + '/data/aes/small.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'hc', [
#         '-i', os.getcwd() + '/data/aes/medium.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'hc', [
#         '-i', os.getcwd() + '/data/aes/large.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cuda', [
#         '-i', os.getcwd() + '/data/aes/small.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cuda', [
#         '-i', os.getcwd() + '/data/aes/medium.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),
#     ('aes', 'cuda', [
#         '-i', os.getcwd() + '/data/aes/large.data',
#         '-k', os.getcwd() + '/data/aes/key.data'
#     ]),

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

#     ('bs', 'hc', ['-x', '1048576']),
#     ('bs', 'hc', ['-x', '1048576', '--collaborative']),
#     ('bs', 'cuda', ['-x', '1048576']),
#     ('bs', 'cuda', ['-x', '1048576', '--collaborative']),

#     ('ep', 'hc', ['-x', '128', '-m', '20']),
#     ('ep', 'hc', ['-x', '256', '-m', '20']),
#     ('ep', 'hc', ['-x', '512', '-m', '20']),
#     ('ep', 'hc', ['-x', '1024', '-m', '20']),
#     ('ep', 'hc', ['-x', '2048', '-m', '20']),
#     ('ep', 'hc', ['-x', '4096', '-m', '20']),
#     ('ep', 'hc', ['-x', '8192', '-m', '20']),
#     ('ep', 'hc', ['-x', '16384', '-m', '20']),
#     ('ep', 'hc', ['-x', '32768', '-m', '20']),
#     ('ep', 'hc', ['-x', '65536', '-m', '20']),
#     ('ep', 'hc', ['-x', '128', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '256', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '512', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '1024', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '2048', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '4096', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '8192', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '16384', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '32768', '-m', '20', '-c']),
#     ('ep', 'hc', ['-x', '65536', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '128', '-m', '20']),
#     ('ep', 'cuda', ['-x', '256', '-m', '20']),
#     ('ep', 'cuda', ['-x', '512', '-m', '20']),
#     ('ep', 'cuda', ['-x', '1024', '-m', '20']),
#     ('ep', 'cuda', ['-x', '2048', '-m', '20']),
#     ('ep', 'cuda', ['-x', '4096', '-m', '20']),
#     ('ep', 'cuda', ['-x', '8192', '-m', '20']),
#     ('ep', 'cuda', ['-x', '16384', '-m', '20']),
#     ('ep', 'cuda', ['-x', '32768', '-m', '20']),
#     ('ep', 'cuda', ['-x', '65536', '-m', '20']),
#     ('ep', 'cuda', ['-x', '128', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '256', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '512', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '1024', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '2048', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '4096', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '8192', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '16384', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '32768', '-m', '20', '-c']),
#     ('ep', 'cuda', ['-x', '65536', '-m', '20', '-c']),

#     ('fir', 'cl12', []),
#     ('fir', 'cl20', []),
#     ('fir', 'hc', ['-y', '1024', '-x', '1024']),
#     ('fir', 'hc', ['-y', '1024', '-x', '2048']),
#     ('fir', 'hc', ['-y', '1024', '-x', '3072']),
#     ('fir', 'hc', ['-y', '1024', '-x', '4096']),
#     ('fir', 'hc', ['-y', '1024', '-x', '5120']),
#     ('fir', 'hc', ['-y', '1024', '-x', '6144']),
#     ('fir', 'hc', ['-y', '1024', '-x', '7168']),
#     ('fir', 'hc', ['-y', '1024', '-x', '8192']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '1024']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '2048']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '3072']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '4096']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '5120']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '6144']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '7168']),
#     ('fir', 'cuda', ['-y', '1024', '-x', '8192']),
#     ('fir', 'hip', ['-y', '1024', '-x', '1024']),
#     ('fir', 'hip', ['-y', '1024', '-x', '2048']),
#     ('fir', 'hip', ['-y', '1024', '-x', '3072']),
#     ('fir', 'hip', ['-y', '1024', '-x', '4096']),
#     ('fir', 'hip', ['-y', '1024', '-x', '5120']),
#     ('fir', 'hip', ['-y', '1024', '-x', '6144']),
#     ('fir', 'hip', ['-y', '1024', '-x', '7168']),
#     ('fir', 'hip', ['-y', '1024', '-x', '8192']),

#     ('ga', 'hc', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'hc', ['-i', os.getcwd() +
#                   '/data/gene_alignment/medium.data', '--collaborative']),
#     ('ga', 'cuda', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'cuda', ['-i', os.getcwd() +
#                     '/data/gene_alignment/medium.data', '--collaborative']),
#     ('ga', 'hip', ['-i', os.getcwd() + '/data/gene_alignment/medium.data']),
#     ('ga', 'hip', ['-i', os.getcwd() +
#                    '/data/gene_alignment/medium.data', '--collaborative']),

#     ('hist', 'cl12', ['-x', '1048576']),
#     ('hist', 'cl20', ['-x', '1048576']),
#     ('hist', 'hc', ['-x', '1048576']),
#     ('hist', 'cuda', ['-x', '65536']),
#     ('hist', 'cuda', ['-x', '131072']),
#     ('hist', 'cuda', ['-x', '196608']),
#     ('hist', 'cuda', ['-x', '262144']),
#     ('hist', 'cuda', ['-x', '327680']),
#     ('hist', 'cuda', ['-x', '393216']),
#     ('hist', 'cuda', ['-x', '458752']),
#     ('hist', 'cuda', ['-x', '524288']),
#     ('hist', 'hip', ['-x', '65536']),
#     ('hist', 'hip', ['-x', '131072']),
#     ('hist', 'hip', ['-x', '196608']),
#     ('hist', 'hip', ['-x', '262144']),
#     ('hist', 'hip', ['-x', '327680']),
#     ('hist', 'hip', ['-x', '393216']),
#     ('hist', 'hip', ['-x', '458752']),
#     ('hist', 'hip', ['-x', '524288']),


#     ('kmeans', 'cl12', ['-i', os.getcwd() + '/data/kmeans/10000_34.txt']),
#     ('kmeans', 'cl20', ['-i', os.getcwd() + '/data/kmeans/10000_34.txt']),
#     ('kmeans', 'hc', ['-i', os.getcwd() + '/data/kmeans/10000_34.txt']),
#     ('kmeans', 'cuda', ['-i', os.getcwd() + '/data/kmeans/10000_34.txt']),

#     ('pr', 'cl12', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
#     ('pr', 'cl20', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
#     ('pr', 'hc', ['-i', os.getcwd() + '/data/page_rank/medium.data']),
# ]

def main():
    """main function"""
    args = parse_args()
    if not args.skip_compile:
        compile(args)

    benchmarks = []
    setup_benchmarks(benchmarks, args)
    run(benchmarks)


def parse_args():
    """parse user input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compile", action="store_true",
                        help="""
            Setting this argument will skip the compilation process. This is 
            useful if you have the compiled with this script before.
            """)
    parser.add_argument("--cxx", default="g++",
                        help="""
            The compiler to be used to compile the benchmark. 
            """)
    parser.add_argument("--skip-verification", action="store_true",
                        help="""
            Setting this argument will skip the CPU verification process.
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
    parser.add_argument("-i", "--ignore-error", action="store_true",
                        help="""
            Use this option to ignore errors in the compilation and 
            verification process.
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


def run(benchmarks):
    """ Run all benchmarks """
    for benchmark in benchmarks:
        benchmark.run()


if __name__ == "__main__":
    main()
