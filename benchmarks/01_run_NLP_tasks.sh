num_samples=10
results_folder=results
overwrite=1
do_inference=1
post_process=1
re_evaluate=0
script=run_nlp_tasks.py

# model=meta-llama/Llama-2-7b-chat-hf
model=openai/gpt-oss-120b
#model=adamic_graph
model_type=groq
model_judge=anthropic/claude-opus-4.6

lang_list=all
prompt_type_list=google_direct
task_list=all

gpu=1,2

CUDA_VISIBLE_DEVICES=$gpu poetry run python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --model_type $model_type \
    --model_judge $model_judge \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process \
    --re_evaluate $re_evaluate
   



