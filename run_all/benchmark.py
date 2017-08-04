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
        command = " ".join([self.executable_full_path, '-q', '-v'] + args)
        proc = subprocess.Popen(command,
                                cwd=self.cwd, shell=True,
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

        if self.options.full_verification:
            self._verify(args)

        print("Benchmarking", self.executable, *args, sep=' ', end=' ')

        runtime_regex = re.compile(
            r'Run: ((0|[1-9]\d*)?(\.\d+)?(?<=\d)) second')

        perf = []
        for i in range(0, self.options.repeat_time):
            command = " ".join([self.executable_full_path, '-q', '-t'] + args)
            proc = subprocess.Popen(command,
                                    cwd=self.cwd, shell=True,
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


class HistBenchmark(Benchmark):
    """Hist benchmark"""

    def __init__(self, options):
        super(HistBenchmark, self).__init__(options)
        self.benchmark_name = 'hist'
        self.benchmark_platforms = ['cl12', 'cl20', 'hc', 'cuda', 'hip']
        self.verify_run = []
        self.benchmark_runs = [
            ['-x', '65536'],
            ['-x', '131072'],
            ['-x', '262144'],
            ['-x', '524288'],
            ['-x', '1048576'],
            ['-x', '2097152'],
            ['-x', '4194304'],
            ['-x', '8388608'],
            ['-x', '1677216'],
            ['-x', '33554432'],
        ]


class PRBenchmark(Benchmark):
    """KMeans benchmark"""

    def __init__(self, options):
        super(PRBenchmark, self).__init__(options)
        self.benchmark_name = 'pr'
        self.benchmark_platforms = ['cl12', 'cl20', 'hc', 'cuda', 'hip']
        self.verify_run = ['-i', os.getcwd() + '/data/pr/1024.data']
        self.benchmark_runs = [
            ['-i', os.getcwd() + '/data/pr/1024.data'],
            ['-i', os.getcwd() + '/data/pr/2048.data'],
            ['-i', os.getcwd() + '/data/pr/4096.data'],
            ['-i', os.getcwd() + '/data/pr/8192.data'],
            ['-i', os.getcwd() + '/data/pr/16384.data'],
        ]


class KMeansBenchmark(Benchmark):
    """KMeans benchmark"""

    def __init__(self, options):
        super(KMeansBenchmark, self).__init__(options)
        self.benchmark_name = 'kmeans'
        self.benchmark_platforms = ['cl12', 'cl20', 'hc', 'cuda', 'hip']
        self.verify_run = ['-i', os.getcwd() + '/data/kmeans/1000_34.txt']
        self.benchmark_runs = [
            ['-i', os.getcwd() + '/data/kmeans/100_34.txt'],
            ['-i', os.getcwd() + '/data/kmeans/1000_34.txt'],
            ['-i', os.getcwd() + '/data/kmeans/10000_34.txt'],
            ['-i', os.getcwd() + '/data/kmeans/100000_34.txt'],
            ['-i', os.getcwd() + '/data/kmeans/1000000_34.txt'],
            ['-i', os.getcwd() + '/data/kmeans/1000000_34.txt'],
        ]


class BSBenchmark(Benchmark):
    """BS benchmark"""

    def __init__(self, options):
        super(BSBenchmark, self).__init__(options)
        self.benchmark_name = 'bs'
        self.benchmark_platforms = ['hc', 'cuda', 'hip']
        self.verify_run = []
        self.benchmark_runs = [
            ['-x', '131072'],
            ['-x', '262144'],
            ['-x', '524288'],
            ['-x', '1048576'],
            ['-x', '2097152'],
            ['-x', '4194304'],
            ['-x', '8388608'],
            ['-x', '131072', '-c', '--chunk', '4096'],
            ['-x', '262144', '-c', '--chunk', '4096'],
            ['-x', '524288', '-c', '--chunk', '4096'],
            ['-x', '1048576', '-c', '--chunk', '4096'],
            ['-x', '2097152', '-c', '--chunk', '4096'],
            ['-x', '4194304', '-c', '--chunk', '4096'],
            ['-x', '8388608', '-c', '--chunk', '4096'],
        ]


class EPBenchmark(Benchmark):
    """EP benchmark"""

    def __init__(self, options):
        super(EPBenchmark, self).__init__(options)
        self.benchmark_name = 'ep'
        self.benchmark_platforms = ['hc', 'cuda', 'hip']
        self.verify_run = []
        self.benchmark_runs = [
            ['-x', '1024', '-m', '20'],
            ['-x', '2048', '-m', '20'],
            ['-x', '4096', '-m', '20'],
            ['-x', '8192', '-m', '20'],
            ['-x', '16384', '-m', '20'],
            ['-x', '32768', '-m', '20'],
            ['-x', '1024', '-m', '20', '-c'],
            ['-x', '2048', '-m', '20', '-c'],
            ['-x', '4096', '-m', '20', '-c'],
            ['-x', '8192', '-m', '20', '-c'],
            ['-x', '16384', '-m', '20', '-c'],
            ['-x', '32768', '-m', '20', '-c'],
        ]


class BEBenchmark(Benchmark):
    """BE benchmark"""

    def __init__(self, options):
        super(BEBenchmark, self).__init__(options)
        self.benchmark_name = 'be'
        self.benchmark_platforms = ['hc', 'cuda', 'hip']
        self.verify_run = ['-i', os.getcwd() + '/data/be/320x180.mp4']
        self.benchmark_runs = [
            ['-i', os.getcwd() + '/data/be/320x180.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/480x270.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/640x360.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/800x450.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/960x540.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1120x630.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1280x720.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1440x810.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1600x900.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1760x990.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/1920x1080.mp4', '-m', '100'],
            ['-i', os.getcwd() + '/data/be/320x180.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/480x270.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/640x360.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/800x450.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/960x540.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1120x630.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1280x720.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1440x810.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1600x900.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1760x990.mp4', '-m', '100', '-c'],
            ['-i', os.getcwd() + '/data/be/1920x1080.mp4', '-m', '100', '-c'],
        ]


class GABenchmark(Benchmark):
    """GA benchmark"""

    def __init__(self, options):
        super(GABenchmark, self).__init__(options)
        self.benchmark_name = 'ga'
        self.benchmark_platforms = ['hc', 'cuda', 'hip']
        self.verify_run = ['-i', os.getcwd() + '/data/ga/1024_64.data']
        self.benchmark_runs = [
            ['-i', os.getcwd() + '/data/ga/1024_64.data'],
            ['-i', os.getcwd() + '/data/ga/2048_128.data'],
            ['-i', os.getcwd() + '/data/ga/4096_256.data'],
            ['-i', os.getcwd() + '/data/ga/8192_512.data'],
            ['-i', os.getcwd() + '/data/ga/16384_1024.data'],
            ['-i', os.getcwd() + '/data/ga/32768_1024.data'],
            ['-i', os.getcwd() + '/data/ga/65536_1024.data'],
            ['-i', os.getcwd() + '/data/ga/131072_1024.data'],
            ['-i', os.getcwd() + '/data/ga/262144_1024.data'],
            ['-i', os.getcwd() + '/data/ga/524288_1024.data'],
            ['-i', os.getcwd() + '/data/ga/1048576_1024.data'],
            ['-i', os.getcwd() + '/data/ga/1024_64.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/2048_128.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/4096_256.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/8192_512.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/16384_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/32768_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/65536_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/131072_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/262144_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/524288_1024.data', '-c'],
            ['-i', os.getcwd() + '/data/ga/1048576_1024.data', '-c'],
        ]
