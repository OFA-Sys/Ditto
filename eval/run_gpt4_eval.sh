export OPENAI_API_KEY=your-openai-api-key
DATA=wiki_roleplay_multilingual_test_input_w_evidence_qwen-72B-chat_qwen-72B-chat_data
INPUT_FILE=../data/results/${DATA}.jsonl
MODEL=gpt-4-1106-preview
TASK=${DATA}_${MODEL}_test
LOG_FILE=log/$TASK.log
LIMIT=0
CALL_PER_MINUTE=10

nohup python -u call_openai_api_gpt4_eval.py \
    --task  $TASK\
    --model $MODEL \
    --input_file $INPUT_FILE \
    --call_per_minute $CALL_PER_MINUTE \
    --temperature 0.2 \
    --n 3 \
    --limit $LIMIT >> $LOG_FILE &
