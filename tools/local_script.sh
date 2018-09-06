KEY_PEM_DIR=/home/hwang/My_Code/AWS/HongyiWKeyPair.pem
KEY_PEM_NAME=HongyiWKeyPair.pem
PUB_IP_ADDR="$1"
echo "Public address of master node: ${PUB_IP_ADDR}"

ssh -o "StrictHostKeyChecking no" ubuntu@${PUB_IP_ADDR}
scp -i ${KEY_PEM_DIR} ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR}:~/.ssh
scp -i ${KEY_PEM_DIR} hosts hosts_address config ubuntu@${PUB_IP_ADDR}:~/
scp -i ${KEY_PEM_DIR} -r ~/My_Code/grad_lossy_compression ubuntu@${PUB_IP_ADDR}:~/
ssh -i ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR} 'sudo apt-get update; cp /home/ubuntu/grad_lossy_compression/tools/remote_script.sh ~/'