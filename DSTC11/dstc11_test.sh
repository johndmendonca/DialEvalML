python DSTC11_score.py \
    --task 1 \
    --xlm_turn DSTC_11_Track_4/eval/task1/encoder_turn.csv \
    --gpt_turn DSTC_11_Track_4/eval/task1/gpt_turn.json \
    --gpt_dial DSTC_11_Track_4/eval/task1/gpt_dial.json \
    --dev_dir logs/FINAL_dev/ \
    --turn_csv DSTC_11_Track_4/eval/task1/dstc11_multilingual_test-turn.csv \
    --dial_csv DSTC_11_Track_4/eval/task1/dstc11_multilingual_test-dial.csv \
    --weights weights/weights_all_crs.json