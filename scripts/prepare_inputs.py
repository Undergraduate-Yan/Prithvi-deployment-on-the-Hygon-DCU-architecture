from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'flood': 'src/preprocessing/prepare_flood_inputs.py',
              'cloud': 'src/preprocessing/prepare_cloud_local.py'})
