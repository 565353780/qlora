cd ../qlr
python qlora.py \
	--model_name_or_path /home/chli/github/textgen/models/llama-2-7b-chat-hf/ \
	--dataset /home/chli/chLi/Sql-Dataset/data_test1_train.json \
	--dataset_formar alpaca \
	--learning_rate 1e-4 \
	--save_steps 10 \
	--save_total_limit 100
