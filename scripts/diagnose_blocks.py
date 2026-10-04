from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'run': 'src/diagnostics/diagnose_blocks.py',
              'aggregate': 'src/diagnostics/aggregate_blocks.py',
              'instrument': 'src/diagnostics/instrument_intermediates.py',
              'compare': 'src/diagnostics/compare_providers.py',
              'cloud-split': 'src/diagnostics/compare_cloud_split.py'})
