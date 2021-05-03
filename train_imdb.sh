#work_path="/global_fs/00_Code/nlp_course/TextClassificationBenchmark/"
work_path="."

out_path="log_imdb/bilstm_ascc_clean1_kl4"
mkdir $work_path/$out_path
export CUDA_VISIBLE_DEVICES=0
nohup python -u $work_path/train_imdb_ascc.py  --synonyms_from_file true --learning_rate 0.005 --dataset imdb --model bilstm_adv --weight_clean 1 --weight_kl 4 --work_path $work_path  --out_path $out_path > $work_path/$out_path/nohup.log 2>&1 &

out_path="log_imdb/cnn_ascc_clean1_kl4"
mkdir $work_path/$out_path
export CUDA_VISIBLE_DEVICES=1
nohup python -u $work_path/train_imdb_ascc.py  --synonyms_from_file true --learning_rate 0.005 --dataset imdb --model cnn_adv    --weight_clean 1 --weight_kl 4 --work_path $work_path  --out_path $out_path > $work_path/$out_path/nohup.log 2>&1 &


# Attention: When first run set --synonyms_from_file false 