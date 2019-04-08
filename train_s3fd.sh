env

export PATH="/data/anaconda3/bin:$PATH"
export TORCH_HOME='/data/.torch/'

cd /data/task/Detection/dev/FaceBoxes.PyTorch-master

python train_s3fd.py --ngpu 8 --save_folder weights/S3FD/ --num_workers 16 --batch_size 64