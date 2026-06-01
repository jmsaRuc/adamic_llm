num_samples=100
results_folder=results-main

overwrite=0
overwrite_judge=0
do_inference=0
post_process=1
re_evaluate=0
script=run_shareGPT.py
tensor_parallel_size=1

#model=meta-llama/llama-4-scout-17b-16e-instruct
#model=adamic_graph/llama-4-scout-17b-16e-instruct
#model=adamic_graph_self_translate/llama-4-scout-17b-16e-instruct
#model=openai/gpt-oss-120b
#model=adamic_graph/gpt-oss-120b
#model=anthropic/claude-opus-4.7
#model=adamic_graph/claude-opus-4.7
#model=deepseek/deepseek-v4-pro
#model=adamic_graph/deepseek-v4-pro
#model=adamic_graph_self_translate/deepseek-v4-pro
#model=adamic_graph/claude-opus-4.7
#model=adamic_graph_self_translate/gpt-oss-120b

model=adamic_graph_self_translate/claude-opus-4.7

#max_tokens=8192
max_tokens=32768

model_type=open_router

#model_judge=anthropic/claude-opus-4.7
model_judge=google/gemini-3.5-flash
#model_judge=openai/gpt-5.5

task_list=shareGPT


#lang_list=all
#prompt_type_list=direct
prompt_type_list=adamic_self_trans
#prompt_type_list=adamic
#prompt_type_list="direct,google"

#prompt_type_list="direct"

#lang_list="ja"
#lang_list="es"
#lang_list="fr"
lang_list="vi"
#lang_list="id"
#lang_list="ko"
#lang_list="zh"
#lang_list="ro"
#lang_list="uk"
#lang_list="no"

gpu=3
CUDA_VISIBLE_DEVICES=$gpu uv run python ./scripts/$script \
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
