num_samples=10
results_folder=results2

overwrite=0
overwrite_judge=0
do_inference=1
post_process=1
re_evaluate=0
script=run_shareGPT.py
tensor_parallel_size=1

#model=meta-llama/llama-4-scout-17b-16e-instruct
#model=adamic_graph/llama-4-scout-17b-16e-instruct
#model=openai/gpt-oss-120b
model=adamic_graph/gpt-oss-120b
#model=anthropic/claude-opus-4.7
#model=adamic_graph/claude-opus-4.7
max_tokens=32768
model_type=groq
model_judge=anthropic/claude-opus-4.7
task_list=shareGPT

lang_list=all
prompt_type_list=adamic
#prompt_type_list="direct,google"

# lang_list=zh
# prompt_type_list=direct


gpu=3
CUDA_VISIBLE_DEVICES=$gpu poetry run python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --model_type $model_type \
    --model_judge $model_judge \
    --max_tokens $max_tokens \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --overwrite_judge $overwrite_judge \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process
