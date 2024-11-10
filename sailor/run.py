import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Agent')
    parser.add_argument('--grpc_port', type=int,
                        help='Port to start grpc server', required=True)
    parser.add_argument('--training_master_port', type=int,
                        help='Port used for training', required=True)
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to the YAML or python config file")
    parser.add_argument('--world_size', type=int,
                        help='world_size (in number of nodes)', required=True)
    args = parser.parse_args()

    os.system(f"python /workspace/nanotron/sailor/run_ft_train.py --grpc_port {args.grpc_port} --config-file {args.config_file} &")
    node_id = os.environ["SLURM_NODEID"]
    if node_id=="0":
        print("Start controller")
        os.system(f"python /workspace/nanotron/sailor/controller.py --grpc_port {args.grpc_port} --training_master_port {args.training_master_port} --world_size {args.world_size} &")
