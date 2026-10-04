from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'flood': 'src/model_export/export_flood_cli.py',
              'cloud': 'src/model_export/export_cloud_onnx.py'})
