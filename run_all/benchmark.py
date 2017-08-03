"""Benchmark provides definitions about all the benchmarks that can be run,
and how to run those benchmarks.
"""
from __future__ import print_function

import os
import subprocess
import re
import sys
import numpy as np
from bcolors import bcolors


class Benchmark(object):
    """Benchmark defines how to run a benchmark
    """

    def __init__(self, options):
        self.options = options
        self.benchmark_name = ''
        self.benchmark_platforms = []
        self.verify_run = []
        self.benchmark_runs = []

        self.avg_times = []
        self.std_devs = []

        self.executable = ''
        self.cwd = ''
        self.executable_full_path = ''

    def run(self):
        """ runs the benchmark, report benchmarking result
        """
        for platform in self.benchmark_platforms:
            self._reset()
            self._define_current_executing(platform)

            if not self._is_executable_found(platform):
                continue

            if not self._verify(self.verify_run):
                continue

            self._benchmark()
            self._report()

    def _reset(self):
        self.avg_times = []
        self.std_devs = []

    def _define_current_executing(self, platform):
        self.executable = self.benchmark_name + "_" + platform
        self.cwd = self.options.build_folder + 'src/' + \
            self.benchmark_name + '/' + platform + '/'
        self.executable_full_path = self.cwd + self.executable

    def _is_executable_found(self, platform):
        if not os.path.isfile(self.executable_full_path):
            print(self.executable_full_path, 'not found, skip.')
            return False
        return True

    def _verify(self, args):
        print("Verifying", self.executable, *args, sep=' ', end=' ')
        proc = subprocess.Popen([self.executable_full_path, '-q', '-v'] + args,
                                cwd=self.cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print(bcolors.FAIL, "error: ", self.executable, bcolors.ENDC,
                  sep='')
            print(proc.communicate())
            return False

        print(bcolors.OKGREEN, "Passed", bcolors.ENDC, sep='')
        return True

    def _benchmark(self):
        for run in self.benchmark_runs:
            self._benchmark_specific_input(run)

    def _benchmark_specific_input(self, args):
        print("Benchmarking", self.executable, *args, sep=' ', end=' ')

        runtime_regex = re.compile(
            r'Run: ((0|[1-9]\d*)?(\.\d+)?(?<=\d)) second')

        perf = []
        for i in range(0, self.options.repeat_time):
            proc = subprocess.Popen([self.executable_full_path, '-q', '-t'] + args,
                                    cwd=self.cwd,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in proc.stderr:
                res = runtime_regex.search(line)
                if res:
                    perf.append(float(res.group(1)))
            print(".", end='')
            sys.stdout.flush()
        print("")

        self.avg_times.append(np.mean(perf))
        self.std_devs.append(np.std(perf))

    def _report(self):
        print("Benchmark ", self.benchmark_name, "results:",)
        print("\ttime: ", self.avg_times)
        print("\tstd_dev: ", self.std_devs)


class FirBenchmark(Benchmark):
    """FIR benchmark"""

    def __init__(self, options):
        super(FirBenchmark, self).__init__(options)
        self.benchmark_name = 'fir'
        self.benchmark_platforms = ['cl12', 'cl20', 'hc', 'cuda', 'hip']
        self.verify_run = []
        self.benchmark_runs = [
            ['-y', '1024', '-x', '1024'],
            ['-y', '1024', '-x', '2048'],
            ['-y', '1024', '-x', '3072'],
            ['-y', '1024', '-x', '4096'],
            ['-y', '1024', '-x', '5120'],
            ['-y', '1024', '-x', '6144'],
            ['-y', '1024', '-x', '7168'],
            ['-y', '1024', '-x', '8192'],
        ]


class AesBenchmark(Benchmark):
    """AES benchmark"""

    def __init__(self, options):
        super(AesBenchmark, self).__init__(options)
        self.benchmark_name = 'aes'
        self.benchmark_platforms = ['cl12', 'cl20', 'hc', 'cuda', 'hip']
        self.verify_run = [
            '-i', os.getcwd() + '/data/aes/1KB.data',
            '-k', os.getcwd() + '/data/aes/key.data'
        ]
        self.benchmark_runs = [
            [
                '-i', os.getcwd() + '/data/aes/1KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/2KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/4KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/8KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/16KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/32KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/64KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/128KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/256KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/512KB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/1MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/2MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/4MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/8MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/16MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
            [
                '-i', os.getcwd() + '/data/aes/32MB.data',
                '-k', os.getcwd() + '/data/aes/key.data'
            ],
        ]
