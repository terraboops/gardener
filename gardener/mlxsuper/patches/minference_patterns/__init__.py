# Package marker for MInference calibration pattern tables.
#
# CRITICAL: Pattern tables shipped here are SYNTHETIC PLACEHOLDERS
# (programmatic defaults, NOT measured from real attention maps).
# See each JSON file's "note" field for the sentinel string.
# load_pattern_table() in minference_prefill.py emits a loud WARNING
# when it detects the synthetic marker.
#
# To generate a real table:
#   scripts/minference_calibrate.py --model <hf-model-id>
