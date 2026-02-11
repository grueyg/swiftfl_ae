set -e

cd ..

# echo "Run distributed mode with ConvNet-2 on FEMNIST..."

### server
# python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_femnist_server.yaml &
# sleep 2

# clients
/home/app/anaconda3/envs/fs/bin/python /home/app/anaconda3/envs/fs/lib/python3.9/site-packages/federatedscope/main.py --cfg /home/app/anaconda3/envs/fs/lib/python3.9/site-packages/federatedscope/test/distributed_femnist_client_1_copy.yaml &
sleep 2
/home/app/anaconda3/envs/fs/bin/python /home/app/anaconda3/envs/fs/lib/python3.9/site-packages/federatedscope/main.py --cfg /home/app/anaconda3/envs/fs/lib/python3.9/site-packages/federatedscope/test/distributed_femnist_client_2_copy.yaml &
# sleep 2
# python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_femnist_client_3.yaml &

