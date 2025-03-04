

source activate spam
cd ..

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --eval_every 1000 \
    --save_every 100000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.0004 \
    --warmup_steps 2000 \
    --num_training_steps 20000 \
    --optimizer stablespam \
    --weight_quant \
    --simulation \
    --weight_group_size 256 \
    --weight_bits 4 \
    --weight_decay 0 \
    --project stablespam \
    --name stablespam_350_fp4_500_0.9_0.7_4e-4 \
    --save_dir saved \
    --restore_optimizer \
    --real_fp4 \
    --gamma1 0.7 \
    --gamma2 0.9 \
    --gamma3 0.999 \
    --update_proj_gap 500 
