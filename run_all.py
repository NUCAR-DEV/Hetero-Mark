#!/usr/bin/env python

"""This file automatically runs all the benchmarks of Heter-Mark
"""

import os
import re
import subprocess
import numpy as np

build_folder = os.getcwd() + '/build-auto-run/'
benchmark_repeat_time = 20
benchmarks = [
    ('fir', 'cl12', []), 
    ('fir', 'cl20', []), 
    ('fir', 'hc', []), 
]

def main():
    compile()
    verify()
    benchmark()

def compile():
    print "Compiling benchmark into", build_folder

    subprocess.call(['rm', '-rf', build_folder])
    subprocess.call(['mkdir', build_folder])

    p = subprocess.Popen(['cmake', os.getcwd()],
        cwd=build_folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    p = subprocess.Popen(['make', '-j80'],
        cwd=build_folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

    print "Compile completed."

def verify():
    for benchmark in benchmarks:
        executable_name = benchmark[0] + '_' + benchmark[1]
        cwd = build_folder + 'src/' + benchmark[0] + '/' + benchmark[1] + '/'
        executable_full_path = cwd + executable_name
    
        if not os.path.isfile(executable_full_path):
            print executable_name, 'not found, skip.'
            continue;

        p = subprocess.Popen([executable_full_path, '-q', '-v'] + benchmark[2], 
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()

def benchmark():
    runtime_regex = re.compile(r'Run: ((0|[1-9]\d*)?(\.\d+)?(?<=\d)) second')

    for benchmark in benchmarks:
        executable_name = benchmark[0] + '_' + benchmark[1]
        cwd = build_folder + 'src/' + benchmark[0] + '/' + benchmark[1] + '/'
        executable_full_path = cwd + executable_name
    
        if not os.path.isfile(executable_full_path):
            print executable_name, 'not found, skip.'
            continue;

        perf = [] 
        for i in range(0, benchmark_repeat_time):
            p = subprocess.Popen([executable_full_path, '-q', '-t'] + benchmark[2],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in p.stderr:
                res = runtime_regex.search(line)
                if res:
                    perf.append(float(res.group(1)))

        print executable_name, np.mean(perf), np.std(perf)
                 


if __name__ == "__main__":
    main()
