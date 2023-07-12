export TASK_NAME=vsp
export MODEL_NAME=xlm-roberta-large

python DialEvalML/train_res_metric.py \
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
	--save_total_limit 2 \
	--load_best_model_at_end=True \
	--evaluation_strategy steps \
	--eval_steps 1000 \
	--save_steps 1000 \
	--save_strategy steps \
	--train_file data/training_data/en_u_train_concat_least.csv \
	--validation_file data/training_data/en_u_val_concat_least.csv \
	--test_file ../data/azure/DAILYD/DAILYD/ml/ml_u_test.csv \
	--label_names labels  