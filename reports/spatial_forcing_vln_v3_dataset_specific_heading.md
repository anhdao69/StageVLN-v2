# SpatialForcing-VLN v3 dataset-specific heading supervision

The complete implementation, debugging, validation, cache, runtime, and smoke
training report is maintained at
[`debug/v3_mixed_r2r_rxr_heading_debug.md`](../debug/v3_mixed_r2r_rxr_heading_debug.md).

Key result: schema-v2 heading supervision uses 15-degree R2R turns and
30-degree RxR turns per record, rejects old global-15-degree caches, and passed
a four-H100 global-batch-64 end-to-end smoke run with all expected gradients.
