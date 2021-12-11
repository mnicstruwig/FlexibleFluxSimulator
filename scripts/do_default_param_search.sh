#!/usr/bin/env bash

echo 'Executing parameter search for reference devices...'
python do_param_search.py \
    --configs \
    '../data/reference_devices/device_a.json' \
    '../data/reference_devices/device_b.json' \
    '../data/reference_devices/device_c.json' \
    '../data/reference_devices/device_d.json' \
    --sample_dirs \
    '../data/2019-05-23/A/' \
    '../data/2019-05-23/B/' \
    '../data/2019-05-23/C/' \
    '../data/2021-03-05/D/' \
    --budget 500
