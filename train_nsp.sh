export TASK_NAME=nsp
export MODEL_NAME=xlm-roberta-large

python DialEvalML/train_ctxres_metric.py \
	--model_name_or_path $MODEL_NAME \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--do_predict \
	--max_seq_length 258 \
	--per_device_train_batch_size 16 \
	--learning_rate 3e-6 \
	--num_train_epochs 3.0 \
	--output_dir DSTC11/exp/xlm-roberta-large/$TASK_NAME/paL \
	--overwrite_output_dir \
	--overwrite_cache \
	--save_total_limit 2 \
	--load_best_model_at_end \
	--evaluation_strategy steps \
	--eval_steps 10000 \
	--save_steps 10000 \
	--resume_from_checkpoint True \
	--save_strategy steps \
	--train_file data/training_data/en_s_train_concat_least.csv \
	--validation_file data/training_data/en_s_val_concat_least.csv \
	--test_file /home/jmen/data/azure/DAILYD/DAILYD/en/en_s_test.csv \
	--label_names labels
