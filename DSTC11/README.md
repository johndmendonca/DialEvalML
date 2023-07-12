# DialEvalML@DSTC11

This folder implements the competition code for our submission to the DSTC11 Track 4 *Robust and Multilingual Automatic Evaluation Metrics for Open-Domain Dialogue Systems*

## Model selection

We train several encoder models for each submetric, ranging from English only, to multilingual and siamese. 

## Development predictions and scoring

Use `dev_predict.py` to obtain development set predictions and correlation scores, for example:

~~~
CUDA_VISIBLE_DEVICES=0 python dev_predict.py --options all --langs all --predict --eval
~~~

## Test set predictions

To obtain test set predictions, select the desired weights and run `dstc11_test.sh` or use `test_predict.py` directly:

~~~
CUDA_VISIBLE_DEVICES=0 python DSTC11_score.py \
    --task 1 \
    --xlm_turn DSTC_11_Track_4/eval/task1/encoder_turn.csv \
    --gpt_turn DSTC_11_Track_4/eval/task1/gpt_turn.json \
    --gpt_dial DSTC_11_Track_4/eval/task1/gpt_dial.json \
    --dev_dir logs/FINAL_dev/ \
    --turn_csv DSTC_11_Track_4/eval/task1/dstc11_multilingual_test-turn.csv \
    --dial_csv DSTC_11_Track_4/eval/task1/dstc11_multilingual_test-dial.csv \
    --weights weights/weights_all_crs.json
~~~

