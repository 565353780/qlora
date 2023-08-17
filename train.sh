cd ../qlr
python qlora.py \
	--model_name_or_path /home/chli/github/textgen/models/chinese-alpaca-2-7b/ \
	--dataset /home/chli/chLi/Sql-Dataset/data_test1_train.json \
	--dataset_formar alpaca \
	--learning_rate 2e-4 \
	--save_steps 1000 \
	--save_total_limit 100 \
	--max_steps 1000000
