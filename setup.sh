cd ..
git clone https://github.com/artidoro/qlora.git qlr

cd qlr
git pull

pip install -U torch torchvision torchaudio
pip install -U -r requirements.txt

pip install -U bitsandbytes
# pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git
