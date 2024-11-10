import grpc
import os
import time
import socket
import argparse
from concurrent import futures
import signal
import argparse
from typing import Dict, cast

from torch import multiprocessing

from orchestration_pb2_grpc import WorkerAgentServicer, add_WorkerAgentServicer_to_server
from orchestration_pb2 import CheckReadyResponse, KillResponse, WorkerConfigurationResponse


class ElasticWorkerAgent(WorkerAgentServicer):
    def __init__(self, script_args):
        self.training_process_alive = False
        self.hostname = socket.gethostname()
        self.gpus_per_node = 4
        self.world_size = 0
        self.node_rank = -1
        self.master_addr = None
        self.script_args = script_args
        print(f"Hello from grpc server {self.hostname}")

    def CheckReady(self, request, context):
        return CheckReadyResponse(is_ready=True)

    def Kill(self, request, context):
        print(f"Killing local process ...")
        if self.training_process_alive:
           print("HERE")
           os.system("pkill -f run_train_custom.py") # TODO: check cleanup
        self.training_process_alive = False
        # TODO: check abort
        return KillResponse()

    def ConfigurationChange(self, request, context):
        assert not self.training_process_alive
        print(f"Got topology: {request.topology}")

        # check if rank in participants
        topology_list = list(request.topology)
        if self.is_in_topo(topology_list):
            print(f"Starting new process, node rank is {self.node_rank}")
            start_cmd_base = f"python run_train_custom.py --config-file {self.script_args.config_file} --world-size {self.world_size} --master-ip {self.master_addr} --master-port {request.port}"
            for i in range(self.gpus_per_node):
                print(f"Start for process {i}")
                rank_i = self.node_rank*self.gpus_per_node + i
                start_cmd_i = start_cmd_base + f" --rank {rank_i} &"
                os.system(start_cmd_i)
            self.training_process_alive = True
        return WorkerConfigurationResponse()

    def is_in_topo(self, topology):
        if self.hostname not in topology:
            return False
        self.node_rank = topology.index(self.hostname)
        self.world_size = len(topology) * self.gpus_per_node
        self.master_addr = topology[0]
        return True

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Arguments for Agent')
    parser.add_argument('--grpc_port', type=int,
                        help='Port to start grpc server', required=True)
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to the YAML or python config file")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent = ElasticWorkerAgent(args)
    add_WorkerAgentServicer_to_server(agent, server)
    server.add_insecure_port(f'[::]:{args.grpc_port}')

    def terminate(signum, _):
        if not agent.training_process_alive:
            os.system("pkill -f run_train_custom.py")
        done = server.stop(5)
        done.wait()
        print(f"Received {signum}, stop complete!")

    print("Start server!")
    server.start()
    signal.signal(signal.SIGTERM, terminate)
    server.wait_for_termination()
