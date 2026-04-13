num_samples=3
results_folder=results

overwrite=0
overwrite_judge=0
do_inference=1
post_process=1
re_evaluate=0
script=run_shareGPT.py
tensor_parallel_size=1
# model=gpt-4.1-nano
model=openai/gpt-oss-120b
model_type=groq
model_judge=anthropic/claude-opus-4.6
task_list=shareGPT

lang_list=all
prompt_type_list=all

# lang_list=zh
# prompt_type_list=direct

# for model_judge in "anthropic/claude-3.5-sonnet" "google/gemini-pro-1.5"; do
for prompt_type_list in direct google; do

gpu=3
CUDA_VISIBLE_DEVICES=$gpu poetry run python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --model_type $model_type \
    --model_judge $model_judge \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --overwrite_judge $overwrite_judge \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process

done 
# done
