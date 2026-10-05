from _entrypoint import dispatch
if __name__ == "__main__":
    dispatch({'clean89': 'src/statistics/clean89.py', 'margins': 'src/statistics/margin_sensitivity.py', 'cloud-confusion': 'src/statistics/cloud_confusion.py', 'process-latency': 'src/statistics/process_latency_ci.py'})
