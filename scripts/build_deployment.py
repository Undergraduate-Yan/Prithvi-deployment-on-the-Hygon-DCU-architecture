from _entrypoint import dispatch
if __name__ == '__main__':
    dispatch({'precision-map': 'src/precision/generate_precision_maps.py',
              'fp16-segments': 'src/precision/build_fp16_segments.py',
              'fp16-b': 'src/precision/build_fp16_b.py',
              'fp16-c': 'src/precision/build_fp16_c.py',
              'flood-graph': 'src/graph/rcs13_builder.py',
              'cloud-graph': 'src/graph/partition_cloud.py',
              'cloud-split': 'src/graph/split_cloud_decoder.py',
              'cloud-quantize': 'src/precision/quantize_cloud.py',
              'cloud-mixed': 'src/precision/assemble_cloud_mixed.py',
              'compile-flood': 'src/runtime/compile_flood_session.py',
              'compile-cloud': 'src/runtime/compile_cloud_session.py'})
