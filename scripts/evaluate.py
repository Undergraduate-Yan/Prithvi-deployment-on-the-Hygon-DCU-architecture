from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'flood': 'src/evaluation/evaluate_flood.py',
              'flood-metrics': 'src/evaluation/flood_metrics.py',
              'cloud': 'src/evaluation/evaluate_cloud.py',
              'cloud-statistics': 'src/statistics/cloud_analysis.py',
              'precision-maps': 'src/precision/evaluate_precision_maps.py'})
