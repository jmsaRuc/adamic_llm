num_samples=2
results_folder=results
overwrite=0
do_inference=1
post_process=1
re_evaluate=0
script=run_nlp_tasks.py

#model=meta-llama/llama-4-scout-17b-16e-instruct
#model=openai/gpt-oss-120b
#model=adamic_graph/llama-4-scout-17b-16e-instruct
#model=adamic_graph/gpt-oss-120b
#model=anthropic/claude-opus-4.7
model=adamic_graph/claude-opus-4.7
model_type=open-router
model_judge=anthropic/claude-opus-4.6

lang_list=all
#prompt_type_list="direct,direct_native,google_direct"
prompt_type_list="adamic"
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
   



