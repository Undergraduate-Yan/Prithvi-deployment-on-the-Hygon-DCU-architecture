from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'python-flood': 'src/benchmarking/benchmark_flood.py',
              'cpp': 'src/benchmarking/run_cpp_processes.py',
              'cloud-trace': 'src/benchmarking/trace_cloud_session.py',
              'compile-memory': 'src/benchmarking/collect_compile_memory.py'})
