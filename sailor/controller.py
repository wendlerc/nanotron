import grpc
import subprocess
import time
import argparse
import os

from orchestration_pb2_grpc import WorkerAgentStub
from orchestration_pb2 import CheckReadyRequest, WorkerConfigurationRequest, KillRequest

def get_slurm_nodelist():
    result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
    hostnames_str = result.stdout.decode("utf-8")
    hostnames_list = hostnames_str.split('\n')
    return hostnames_list[:-1]

class ClusterController:
    def __init__(self, world_size: int, grpc_port: int) -> None:
        self.world_size = world_size
        self.hostnames = get_slurm_nodelist()
        addresses = ",".join(self.hostnames)
        os.environ["no_proxy"] = addresses
        self.num_nodes = len(self.hostnames)
        self.grpc_port = grpc_port

        print(f"Num nodes {self.num_nodes}, Hostnames: {self.hostnames}")
        self.alive_nodes = []


    def monitor(self) -> None:
        while True:
            print("hello!")
            # get current topology
            new_alive_nodes = self.check_ready()
            print(f"new_alive_nodes is {new_alive_nodes}")

            # check for changes
            if new_alive_nodes != self.alive_nodes:
                # change found
                self.alive_nodes = new_alive_nodes

                # kill all
                self.kill_all()

                # get new topology
                new_topology = self.decide_topology()
                if len(new_topology) == 0:
                    print(f"{self.world_size} nodes are needed, but only {len(self.alive_nodes)} are available! Aborting....")
                    self.kill_all(abort=True)
                    break

                # broadcast new topology
                self.send_new_topology(new_topology)
            else:
                self.alive_nodes = new_alive_nodes
            time.sleep(10)


    def decide_topology(self) -> list[str]:
        if len(self.alive_nodes) < self.world_size:
            return []
        else:
            return self.alive_nodes[:self.world_size]


    def send_new_topology(self, topology: list[str]) -> None:
        for node in self.alive_nodes:
            self.send_new_topology_to_node(node, topology)


    def send_new_topology_to_node(self, node: str, topology: list[str]) -> None:
        request = WorkerConfigurationRequest(topology=topology)
        grpc_target = f'{node}:{self.grpc_port}'
        with grpc.insecure_channel(grpc_target) as channel:
            stub = WorkerAgentStub(channel)
            stub.ConfigurationChange(request)


    def check_ready(self) -> list[str]:
        new_alive_nodes = []
        for node in self.hostnames:
            if self.check_ready_node(node):
                new_alive_nodes.append(node)
        return new_alive_nodes


    def check_ready_node(self, node: str) -> bool:
        print(f"Check if node {node} is ready")
        request = CheckReadyRequest()
        grpc_target = f'{node}:{self.grpc_port}'
        try:
            with grpc.insecure_channel(grpc_target) as channel:
                stub = WorkerAgentStub(channel)
                response = stub.CheckReady(request)
            return response.is_ready
        except Exception as e:
            print(e)
            return False

    def kill_all(self, abort=False) -> None:
        for node in self.alive_nodes:
            self.kill_node(node, abort=abort)


    def kill_node(self, node: str, abort: bool) -> None:
        request = KillRequest(abort=abort)
        grpc_target = f'{node}:{self.grpc_port}'
        with grpc.insecure_channel(grpc_target) as channel:
            stub = WorkerAgentStub(channel)
            stub.Kill(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for Controller')
    parser.add_argument('--world_size', type=int,
                        help='world_size (in number of nodes)', required=True)
    parser.add_argument('--grpc_port', type=int,
                        help='Port to start grpc server', required=True)

    args = parser.parse_args()

    time.sleep(10) # some sleep time to allow the workers to start their grpc servers
    controller = ClusterController(args.world_size, args.grpc_port)
    controller.monitor()