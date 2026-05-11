#!/bin/bash
for i in 11 12 13 14 15 16 17; do
    COLLECT_GENERATOR=sinusoidal COLLECT_DURATION=3600 \
    python simulation/scenes/collect_data.py \
        --scenes simulation/configs/generated_scenes.yaml \
        --scene-idx $i \
        --output-dir /media/chen-lab/84BABCB7BABCA6D81/Yifan/sofa_data/generated_scenes &
done
wait
echo "All done"
