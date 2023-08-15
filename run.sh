cd ../qlr
python qlora.py \
	--model_name_or_path /home/chli/github/textgen/models/llama-2-7B-Guanaco-QLoRA-GPTQ/ \
	--dataset /home/chli/github/textgen/training/datasets/data1_train.json \
	--dataset_formar alpaca
