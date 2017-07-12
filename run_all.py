#!/usr/bin/env python

"""This file automatically runs all the benchmarks of Heter-Mark
"""
from __future__ import print_function

import os
import re
import sys
import subprocess
import numpy as np

build_folder = os.getcwd() + '/build-auto-run/'
benchmark_repeat_time = 5
benchmarks = [
    ('aes', 'cl12', [
        '-i', os.getcwd()+'/data/aes/medium.data',
        '-k', os.getcwd() + '/data/aes/key.data'
    ]),
    ('aes', 'cl20', [
        '-i', os.getcwd()+'/data/aes/medium.data',
        '-k', os.getcwd() + '/data/aes/key.data'
    ]),
    ('aes', 'hc', [
        '-i', os.getcwd()+'/data/aes/medium.data',
        '-k', os.getcwd() + '/data/aes/key.data'
    ]),
    ('aes', 'hsa', [
        '-i', os.getcwd()+'/data/aes/medium.data',
        '-k', os.getcwd() + '/data/aes/key.data'
    ]),

    ('fir', 'cl12', []),
    ('fir', 'cl20', []),
    ('fir', 'hc', []),
    ('fir', 'hsa', []),
]
compiler='/opt/rocm/bin/hcc'


def main():
    compile()
    verify()
    benchmark()

def compile():
    compile_log_filename = "compile_log.txt"

    print("Compiling benchmark into", build_folder)

    subprocess.call(['rm', '-rf', build_folder])
    subprocess.call(['mkdir', build_folder])

    env = os.environ.copy()
    env['CXX'] = compiler
    compile_log = open(compile_log_filename, "w")
    p = subprocess.Popen('cmake ' + os.getcwd(),
        cwd=build_folder, env=env, shell=True,
        stdout=compile_log, stderr=compile_log)
    p.wait()
    if p.returncode != 0:
        print(bcolors.FAIL + "Compile failed, see", compile_log_filename,
                "for detailed information", bcolors.ENDC)
        exit(-1)

    p = subprocess.Popen('make VERBOSE=1',
        cwd=build_folder, shell=True, 
        stdout=compile_log, stderr=compile_log)
    p.wait()
    if p.returncode != 0:
        print(bcolors.FAIL + "Compile failed, see", compile_log_filename,
                "for detailed information", bcolors.ENDC)
        exit(-1)

    print(bcolors.OKGREEN + "Compile completed." + bcolors.ENDC)

def verify():
    for benchmark in benchmarks:
        executable_name = benchmark[0] + '_' + benchmark[1]
        cwd = build_folder + 'src/' + benchmark[0] + '/' + benchmark[1] + '/'
        executable_full_path = cwd + executable_name

        if not os.path.isfile(executable_full_path):
            print(executable_name, 'not found, skip.')
            continue;

        p = subprocess.Popen([executable_full_path, '-q', '-v'] + benchmark[2],
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        if p.returncode != 0:
            print(bcolors.FAIL, "error: ", executable_name, bcolors.ENDC, 
		sep='')

def benchmark():
    runtime_regex = re.compile(r'Run: ((0|[1-9]\d*)?(\.\d+)?(?<=\d)) second')

    for benchmark in benchmarks:
        executable_name = benchmark[0] + '_' + benchmark[1]
        cwd = build_folder + 'src/' + benchmark[0] + '/' + benchmark[1] + '/'
        executable_full_path = cwd + executable_name

        if not os.path.isfile(executable_full_path):
            print(executable_name, 'not found, skip.')
            continue;

        print("Benchmarking ", executable_name, ": ", sep='', end='')
        sys.stdout.flush()

        perf = []
        for i in range(0, benchmark_repeat_time):
            p = subprocess.Popen([executable_full_path, '-q', '-t'] + benchmark[2],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in p.stderr:
                res = runtime_regex.search(line)
                if res:
                    perf.append(float(res.group(1)))
            print(".", end=''); sys.stdout.flush()

        print("\n" + executable_name+ ": "
                + str(np.mean(perf)) + ", "
                + str(np.std(perf)))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    main()
