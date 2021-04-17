mpirun -np 17 -host gn10,gn10,gn10,gn10,gn10,gn11,gn11,gn11,gn11,gn12,gn12,gn12,gn12,gn13,gn13,gn13,gn13 \
/THL5/home/daodao/anaconda2/envs/pytorch0.3.0_1/bin/python distributed_nn.py \
--lr=0.8 \
--lr-shrinkage=0.95 \
--momentum=0.9 \
--network=ResNet34_imagenet \
--dataset=Imagenet \
--batch-size=256 \
--test-batch-size=128 \
--comm-type=Bcast \
--num-aggregate=2 \
--eval-freq=250 \
--epochs=50 \
--max-steps=10000000 \
--svd-rank=3 \
--quantization-level=4 \
--bucket-size=512 \
--code=lss_sgd \
--enable-gpu=True\
--train-dir=../
