num_samples=500
results_folder=results-main
overwrite=0
do_inference=1
post_process=1
re_evaluate=0
script=run_nlp_tasks.py

#model=meta-llama/llama-4-scout-17b-16e-instruct
#model=openai/gpt-oss-120b
#model=adamic_graph/llama-4-scout-17b-16e-instruct
#model=adamic_graph_self_translate/llama-4-scout-17b-16e-instruct
#model=adamic_graph/gpt-oss-120b
#model=anthropic/claude-opus-4.7
#model=adamic_graph_self_translate/claude-opus-4.7
#model=adamic_graph_self_translate/gpt-oss-120b
#model=deepseek/deepseek-v4-pro
#model=adamic_graph/deepseek-v4-pro
model=adamic_graph_self_translate/deepseek-v4-pro
model_type=groq
model_judge=anthropic/claude-opus-4.6
#max_tokens=16384
max_tokens=8192

#xl-sum 
#lang_list="fr,zh"
#lang_list="es,vi"
lang_list="tr"

#mkqa
#lang_list="de,ru,fr"
#lang_list="zh_cn,es,ja"
#lang_list="vi,tr,th"

#xnli
#lang_list="de,ru,fr,zh,es"
#lang_list="vi,tr,sw,ar,el"
#lang_list="th,bg,hi,ur"

#paws-x
#lang_list="de,fr"
#lang_list="zh,es"
#lang_list="ja,ko"

#mmmlu
#lang_list="ar,bn,de,es,fr"
#lang_list="hi,id,it,ja,ko"
#lang_list="pt,sw,yo,zh_cn"


#global_mmlu
#lang_list="de,fa,fr,he,ja"
#lang_list="ne,pl,si,sn"
#lang_list="so,sr,sv,yo"


#lang_list=all

prompt_type_list=adamic_self_trans
#prompt_type_list=direct_native
#prompt_type_list=direct
#prompt_type_list="google_direct"
#prompt_type_list="direct_native"
#prompt_type_list="adamic"

#task_list="mkqa,xlsum"
#task_list="mmmlu,paws-x"
#task_list="global_mmlu,mmmlu,paws-x,mkqa,xlsum,xnli"
#task_list="global_mmlu"
#task_list="mmmlu"
#task_list="xnli"
#task_list="paws-x"
#task_list="mkqa"
task_list="xlsum"
gpu=1,2

CUDA_VISIBLE_DEVICES=$gpu uv run python ./scripts/$script \
    --task_list $task_list \
    --model  $model \
    --model_type $model_type \
    --model_judge $model_judge \
    --max_tokens $max_tokens \
    --prompt_type_list $prompt_type_list \
    --lang_list $lang_list \
    --overwrite $overwrite \
    --num_samples $num_samples \
    --results_folder $results_folder \
    --do_inference $do_inference \
    --post_process $post_process \
    --re_evaluate $re_evaluate
   



