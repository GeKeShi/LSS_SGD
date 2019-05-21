mpirun --host gn6,gn7,gn8 \
~/anaconda2/envs/pytorch0.3.0/bin/python distributed_nn.py \
--lr=0.01 \
--lr-shrinkage=0.95 \
--momentum=0.0 \
--network=ResNet18 \
--dataset=Cifar100 \
--batch-size=128 \
--test-batch-size=200 \
--comm-type=Bcast \
--num-aggregate=2 \
--eval-freq=200 \
--epochs=20 \
--max-steps=1000000 \
--svd-rank=3 \
--quantization-level=4 \
--bucket-size=512 \
--code=qsgd \
--enable-gpu=True\
--train-dir=../
