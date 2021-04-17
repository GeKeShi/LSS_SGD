mpirun -np 17 -host gn31,gn31,gn31,gn31,gn31,gn7,gn7,gn7,gn7,gn8,gn8,gn8,gn8,gn9,gn9,gn9,gn9 \
/THL5/home/daodao/anaconda2/envs/pytorch0.3.0_1/bin/python distributed_nn.py \
--lr=0.4 \
--lr-shrinkage=0.95 \
--momentum=0.9 \
--network=ResNet34_imagenet \
--dataset=Imagenet \
--batch-size=128 \
--test-batch-size=128 \
--comm-type=Bcast \
--num-aggregate=2 \
--eval-freq=250 \
--epochs=50 \
--max-steps=10000000 \
--svd-rank=3 \
--quantization-level=4 \
--bucket-size=512 \
--code=lss \
--enable-gpu=True\
--train-dir=../
