# Data, models and external artifacts

## Included records

The repository includes dataset selection identifiers, cloud source-part locations, task contracts, numerical tables, derived scene metrics, cloud scene confusion matrices and model/cache identities. It contains no raw imagery, labels, prediction tensors, model weights, ONNX graphs or MXR binaries.

## Dataset inputs

- **Flood:** acquire Sen1Floods11 through its original distribution and comply with its terms. The source preparation code expects the `v1.1` hand-labelled image/label and split layout. Selection identifiers are in `manifests/datasets/flood/`. The calibration, configuration, exposed public-pool and official validation reserve lists serve different roles.
- **Cloud:** the exact TACO part URLs and dataset revision are recorded in `manifests/datasets/cloud/formal_records.csv`. Obtain the original CloudSEN12+ parts under their applicable terms. `prepare_inputs.py cloud` reads those local parts at the retained byte offsets, applies the original six-band selection, validity masks and 0.0001 scale, and creates new scene payloads. It does not download or unlock an evaluation automatically.

## Models

Prithvi-EO-2.0-300M is the encoder family. The flood and cloud models use different downstream checkpoints and output heads. A pretrained encoder alone is insufficient to reproduce either reported task model.

The flood export adapter expects a supplied study input tree containing:

```text
external/flood/
  data/sen1floods11/v1.1/...
  models/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt
  outputs/fp32_baseline_full/checkpoints/...
  outputs/fp32_baseline_full/... evaluation and checkpoint-selection metadata
```

The retained exporter reads the checkpoint-selection/evaluation metadata through `evaluate_fp32_baseline.py`. Use the exact selected downstream checkpoint; training again is not a substitute for the fixed artifact identity. Graph hashes can change with preparation software or serialization even when weights are unchanged.

## Artifact locations

`manifests/artifacts/required_artifacts.json` lists identities extracted from the task records. Put each required file at:

```text
external/artifacts/<sha256>/<original-basename>
```

Portable flood runtime manifests use `artifact://` paths. Resolve a chosen manifest without modifying its numerical values:

```bash
python scripts/resolve_artifacts.py --manifest configs/flood/runtime/RCS13_FP32.json --artifact-root external/artifacts --output outputs/manifests/flood_fp32.json
```

The resolver checks size and SHA256. It does not create or download missing material. Cloud manifests list the ordered model/cache sessions; supply them in that order to the cloud evaluator. Runtime inputs, compile feeds and construction reports remain separate prerequisites.

Some builders verify the exact bytes of their dependency source and topology manifest. Those files are preserved with Git line-ending conversion disabled. Do not change digest checks to admit a different artifact.

## Access conditions

Study-authored code and derived records not subject to third-party restrictions may be requested reasonably from Jibing Qiu at qiujibing@ict.ac.cn. Dataset imagery, pretrained resources, vendor libraries and other third-party components remain governed by their owners' distribution and licensing terms. Requests do not imply permission to redistribute restricted material.

## Completeness

The aggregate tables and included scene records support offline examination. Full deployment reproduction additionally depends on exact task weights, graph/cache payloads, calibration inputs, construction reports and compatible K100 hardware/software. Their availability must be assessed for each requested experiment; this repository is not a self-contained substitute for those artifacts.
