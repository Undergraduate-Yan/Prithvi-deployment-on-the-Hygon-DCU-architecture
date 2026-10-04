# Configuration scope

The evaluation, precision and timing JSON files describe the paper protocols. They are not interchangeable with runtime manifests. Runtime manifests carry ordered tensor names, model/cache paths and artifact identities required by the selected runner.

`topology/flood_partition_locked.json` is a byte-preserved dependency of the graph builder. Its historical paths are identity metadata, not portable file locations. Supply local model and output paths through the CLI arguments. Do not change its bytes or its digest to make an incompatible graph pass.
