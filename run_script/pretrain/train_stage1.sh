EXPERIMENT_NAME=stage1_xl_prepared_add_mix
DATASET_NAME=vqa
DATA_DIR=./Brote_pretrain/data # dir to brote pretraining data with generated condition context
data_type=pretrain
num_workers=2

model_type=instructblip
model_name_or_path=Salesforce/instructblip-flan-t5-xl
processor_name_or_path=Salesforce/instructblip-flan-t5-xl

bs=8
eval_bs=8
lr=1e-5
dropout=0.1
epoch=4
seed=1234
do_train=True
do_test=False
do_valid=True
master_port=29505

/yeesuanAI05/thumt/wzy/envs/mmicl/bin/deepspeed --master_port $master_port --include localhost:0,1,2,3 run_prepared.py \
    --full_bf16_training True \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --weight_decay 0.0005 \
    --qformer_lr 1e-5 \
    --llm_lr 1e-5 \
    --condition_projection_lr 1e-4 \
    --mix_blip2 True \
    --unfreeze_llm False \
    --unfreeze_qformer True \
    --unfreeze_qtoken True \
    --max_label_length 64 \
    --load_datatype ${data_type} \
    --experiment_name ${EXPERIMENT_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name None \
    --max_seq_length 512 \
    --overwrite_cache False \
    --pad_to_max_length True \
    --train_file ${DATA_DIR} \
    --do_train $do_train \
    --do_eval $do_valid \
    --do_predict $do_test \
    --dataloader_num_workers ${num_workers} \
    --per_device_train_batch_size ${bs} \
    --bf16 \
    --model_type $model_type \
    --save_total_limit 2 \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps 8 \
    --num_train_epochs ${epoch} \
    --output_dir checkpoints/stage1/${EXPERIMENT_NAME} \
    --overwrite_output_dir \
    --seed ${seed} \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --remove_unused_columns False \
    --model_name_or_path $model_name_or_path \
    --processor_path $processor_name_or_path \
    --use_fast_tokenizer True \
    --model_revision main \
    --eval_type val \
    --generation_max_length 256 \
    --using_instruct_qformer False \
    --deepspeed config/deepspeed_config_wo_optimizer.json \
    --run_name condition_add_stage1 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --save_strategy steps \
    --save_steps 500 \
    --dual_loss False \
    --to_wandb False \
    --max_eval_samples 2000 \
    --max_predict_samples 2000 \
    --global_calculation add \
    --logging_steps 1
