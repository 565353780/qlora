cd ../qlr
python qlora.py \
	--model_name_or_path /home/chli/github/textgen/models/llama-2-7b-chat-hf/ \
	--output_dir ./output \
	--do_train False \
	--do_eval False \
	--do_predict True \
	--predict_with_generate \
	--per_device_eval_batch_size 1 \
	--dataset /home/chli/chLi/Sql-Dataset/data_test1_train.json \
	--source_max_len 512 \
	--target_max_len 128 \
	--max_new_tokens 64 \
	--do_sample \
	--top_p 0.9 \
	--num_beams 1
