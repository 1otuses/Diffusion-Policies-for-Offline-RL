export PYTHONPATH=$PYTHONPATH:.

python scripts/eval_saved_model.py \
    --results_dir "results/halfcheetah-medium-v2|halfcheetah_medium|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0" \
    --env_name HalfCheetah-v2 \
    --episodes 5 \
    --render \
    --best \
    --out_dir "results/halfcheetah-medium-v2|halfcheetah_medium|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0/eval_videos" \
    --record