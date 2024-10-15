from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import json
import math

class MVP(object):

    def __init__(self):
        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
        self.set_argparse()
        self.get_json()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="DeepDriveMD_miniapp_EnTK_serial")

        parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases in the workflow')
        parser.add_argument('--num_pipelines', type=int, default=1,
                        help='number of pipelines to launch')
        parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                        help='number of step in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                        help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--conda_env', default=None,
                        help='the conda env where numpy/cupy installed, if not specified, no env will be loaded')
        parser.add_argument('--num_sample', type=int, default=500,
                        help='num of samples in matrix mult (training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                        help='number of matrix mult to perform in training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                        help='time for doing preprocess in agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                        help='number of epochs in agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                        help='number of matrix mult to perform in agent task, inference')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                        help='number of matrix mult to perform in agent task, outlier')
        parser.add_argument('--enable_darshan', action='store_true',
                        help='enable darshan analyze')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=12,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=3,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    # This is for simulation, return a stage which has many sim task
    def run_sim(self, phase_idx, pipeline_idx):
        #phase_cores = {0:1, 1:3, 2:7, 3:1, 4:3, 5:7}
        phase_cores = {0:3}
        s = entk.Stage()
        for i in range(self.args.num_sim):
            t = entk.Task()
            t.pre_exec = [
                "module load DefApps-2023",
                "module load cuda/11.7.1",                
                "module load gcc/9.3.0",
                "module load hdf5/1.14.0",
                "module load python/3.8-anaconda3", 
                'eval "$(conda shell.posix hook)"',
                "conda activate ve.rp",
                'export CUPY_CACHE_DIR="/gpfs/alpine2/scratch/dewiy/gen010/.cupy/kernel_cache"',
                "module unload darshan-runtime"
                ]
            if self.args.conda_env is not None:
                t.pre_exec.append("conda activate {}".format(self.args.conda_env))
            if self.args.enable_darshan:
                t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
            else:
                t.executable = "python"
            t.arguments = ['{}/Executables/simulation.py'.format(self.args.work_dir),
                           '--phase={}'.format(phase_idx),
                           '--pipeline_idx={}'.format(pipeline_idx),
                           '--task_idx={}'.format(i),
                           '--mat_size={}'.format(self.args.mat_size),
                           '--data_root_dir={}'.format(self.args.data_root_dir),
                           '--num_step={}'.format(self.args.num_step),
                           '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["write"]),
                           '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["sim"]["read"])]
            t.post_exec = []
            t.cpu_reqs = {
                 'cpu_processes'    : 1, # number of ranks per task - don't change this no MPI support
                 'cpu_process_type' : None,
                 'cpu_threads'      : phase_cores[phase_idx], # number of ranks so that all cores are used change from 1 to T (where T uses all the resources)
                 'cpu_thread_type'  : rp.OpenMP
                 }
            t.gpu_reqs = {
                 'gpu_processes'     : 1,
                 'gpu_process_type'  : None
                 }
            s.add_tasks(t)

        return s


    # This is for training, return a stage which has a single training task
    def run_train(self, phase_idx, pipeline_idx):

        phase_cores = {0:7, 1:7, 2:7, 3:3, 4:3, 5:3}
        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = [
                "module load DefApps-2023",
                "module load cuda/11.7.1",                
                "module load gcc/9.3.0",
                "module load hdf5/1.14.0",
                "module load python/3.8-anaconda3", 
                'eval "$(conda shell.posix hook)"',
                "conda activate ve.rp",
                'export CUPY_CACHE_DIR="/gpfs/alpine2/scratch/dewiy/gen010/.cupy/kernel_cache"',
                "module unload darshan-runtime"
                ]
        if self.args.conda_env is not None:
                t.pre_exec.append("conda activate {}".format(self.args.conda_env))


        if self.args.enable_darshan:
            t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            t.executable = "python"
        t.arguments = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--split_train={}'.format(0),
                       '--num_epochs={}'.format(self.args.num_epochs_train), # can use this to make training take longer (more epochs)
                       '--device=gpu', # can set this to be cpu if we want but not recommended
                       '--phase={}'.format(phase_idx),
                       '--pipeline_idx={}'.format(pipeline_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       # number of samples changes how much simulation data is used
                       '--num_sample={}'.format(self.args.num_sample * (1 if phase_idx == 0 else 2)),
                       '--num_mult={}'.format(self.args.num_mult_train),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_train),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["read"])]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : phase_cores[phase_idx], # could tune this but most work is done on GPU, maybe reduce to show that we could free up cores for other tasks
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1, # don't increase gpu usage, multi-gpu not supported
            'gpu_process_type'  : None
                }
        s.add_tasks(t)
        # more training tasks is possible but they'd be identical configuration-wise wouldn't speed up workflow, but we could look at when system is fully utilized etc
        # simulation or training tend to take the longest
        return s

    # This is for model selection, return a stage which has a single training task
    def run_selection(self, phase_idx, pipeline_idx):

        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = [
                "module load DefApps-2023",
                "module load cuda/11.7.1",                
                "module load gcc/9.3.0",
                "module load hdf5/1.14.0",
                "module load python/3.8-anaconda3", 
                'eval "$(conda shell.posix hook)"',
                "conda activate ve.rp",
                'export CUPY_CACHE_DIR="/gpfs/alpine2/scratch/dewiy/gen010/.cupy/kernel_cache"',
                "module unload darshan-runtime"
                ]
        if self.args.conda_env is not None:
                t.pre_exec.append("conda activate {}".format(self.args.conda_env))

        if self.args.enable_darshan:
            t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            t.executable = "python"
        t.arguments = ['{}/Executables/selection.py'.format(self.args.work_dir),
                       '--phase={}'.format(phase_idx),
                       '--pipeline_idx={}'.format(pipeline_idx),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["selection"]["read"])]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 7, # can tune this but won't see much effect because it's shortest
            'cpu_thread_type'   : rp.OpenMP
                }
        s.add_tasks(t)

        return s

    # This is for agent, return a stage which has a single training task
    def run_agent(self, phase_idx, pipeline_idx):

        s = entk.Stage()
        t = entk.Task()
        t.pre_exec = [
                "module load DefApps-2023",
                "module load cuda/11.7.1",                
                "module load gcc/9.3.0",
                "module load hdf5/1.14.0",
                "module load python/3.8-anaconda3", 
                'eval "$(conda shell.posix hook)"',
                "conda activate ve.rp",
                'export CUPY_CACHE_DIR="/gpfs/alpine2/scratch/dewiy/gen010/.cupy/kernel_cache"',
                "module unload darshan-runtime"
                ]
        if self.args.conda_env is not None:
                t.pre_exec.append("conda activate {}".format(self.args.conda_env))

        if self.args.enable_darshan:
            t.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            t.executable = "python"
        t.arguments = ['{}/Executables/agent.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_agent),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--pipeline_idx={}'.format(pipeline_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample),
                       '--num_mult={}'.format(self.args.num_mult_agent),
                       '--num_mult_outlier={}'.format(self.args.num_mult_outlier),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_agent),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["agent"]["read"])]
        t.post_exec = []
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : self.args.num_sim, # don't change this, fixed based on 
            'cpu_thread_type'   : rp.OpenMP
                }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : None
                }
        s.add_tasks(t)

        return s
    '''
    def add_soma_service(self):
        t = entk.Task()
        t.pre_exec = [
                    'module reset',
                    "module load DefApps-2023",
                    "module load cuda/11.7.1",
                    "module load gcc/12.1.0",
                    "module load cmake/3.23.2",
                    #"module load cmake/3.27.7",
                    "module load spectrum-mpi/10.4.0.6-20230210",
                    'module load libfabric/1.14.1-sysrdma',
                    'module list',
                    'export SOMA_INSTALL_DIR=/ccs/home/dewiy/soma-collector/install',
                    'export SOMA_NUM_SERVER_INSTANCES=2',
                    'export SOMA_NUM_SERVERS_PER_INSTANCE=64',
                    'export SOMA_SERVER_ADDR_FILE=$RP_PILOT_SANDBOX/server.add',
                    'export SOMA_NODE_ADDR_FILE=$RP_PILOT_SANDBOX/node.add',
                    ]

        t.executable = '/ccs/home/dewiy/soma-collector/build/examples/example-server'
        t.arguments = ['-a', 'ofi+verbs://']
        t.cpu_reqs = {
                'cpu_processes' : 128,
                'cpu_process_type' : None
                }
        t.tags = {'colocate': 'service'}

        # add service task to app manager
        self.am.services = t

    def generate_rp_monitor(self):
        t = entk.Task()
        t.pre_exec = [
                    'module reset',
                    "module load DefApps-2023",
                    "module load cuda/11.7.1",
                    "module load gcc/12.1.0",
                    "module load cmake/3.23.2",
                    #"module load cmake/3.27.7",
                    "module load spectrum-mpi/10.4.0.6-20230210",
                    'module load libfabric/1.14.1-sysrdma',
                    'export SOMA_INSTALL_DIR=/ccs/home/dewiy/soma-collector/install',
                    'export SOMA_NUM_SERVER_INSTANCES=2',
                    'export SOMA_NUM_SERVERS_PER_INSTANCE=64',
                    'export SOMA_MON_SERVER_INSTANCE_ID=0',
                    'export SOMA_SERVER_ADDR_FILE=$RP_PILOT_SANDBOX/server.add',
                    'export SOMA_NODE_ADDR_FILE=$RP_PILOT_SANDBOX/node.add',
                    'export RP_SOMA_MONITORING_FREQUENCY=1',
                    'export RP_WRITE_FREQUENCY=1',
                    'export RP_SOMA_SLEEP_TIME=800000',
                    'export RP_SOMA_RUN_TIME=30',
                    'export RP_FILE_PATH=$RP_PILOT_SANDBOX'
                ]
        t.executable = '/ccs/home/dewiy/soma-collector/build/examples/rp-monitor'
        t.cpu_reqs = {
                'cpu_processes' : 1,
                'cpu_process_type' : None
                }
        t.tags = {'colocate' : 'service'}

        # return the task
        return t
    
    def generate_hw_monitors(self):
        t_list = []
        for idx in range(self.args.num_nodes-8):
            t = entk.Task()
            t.pre_exec = [
                        'module reset',
                        "module load DefApps-2023",
                        "module load cuda/11.7.1",
                        "module load gcc/12.1.0",
                        "module load cmake/3.23.2",
                        #"module load cmake/3.27.7",
                        "module load spectrum-mpi/10.4.0.6-20230210",
                        'module load libfabric/1.14.1-sysrdma',
                        'export SOMA_INSTALL_DIR=/ccs/home/dewiy/soma-collector/install',
                        'export PROC_SOMA_IDX='+str(idx),
                        'export SOMA_NUM_SERVER_INSTANCES=2',
                        'export SOMA_NUM_SERVERS_PER_INSTANCE=64',
                        'export SOMA_MON_SERVER_INSTANCE_ID=1',
                        'export SOMA_SERVER_ADDR_FILE=$RP_PILOT_SANDBOX/server.add',
                        'export SOMA_NODE_ADDR_FILE=$RP_PILOT_SANDBOX/node.add',
                        'export PROC_SOMA_MONITORING_FREQUENCY=1',
                        'export PROC_WRITE_FREQUENCY=20',
                        'export PROC_SOMA_SLEEP_TIME=60000',
                        'export PROC_SOMA_RUN_TIME=25',
                        ]
            t.executable = '/ccs/home/dewiy/soma-collector/build/examples/proc-monitor'
            t.tags = {'colocate' : str(idx), 'exclusive': True}
            t_list.append(t)

        # return the list of tasks
        return t_list

    '''
    def generate_pipeline(self, pipeline_idx):

        p = entk.Pipeline()
        # wait here for monitors to be launched
        t = entk.Task()
        t.executable = 'sleep'
        t.arguments = ['30']
        s = entk.Stage()
        s.add_tasks(t)
        p.add_stages(s)
        for phase in range(int(self.args.num_phases)):
            s1 = self.run_sim(phase, pipeline_idx)
            p.add_stages(s1)
            s2 = self.run_train(phase, pipeline_idx)
            p.add_stages(s2)
            s3 = self.run_selection(phase, pipeline_idx)
            p.add_stages(s3)
            s4 = self.run_agent(phase, pipeline_idx)
            p.add_stages(s4)
        return p
    '''
    def generate_monitor_pipeline(self):
        m = entk.Pipeline()
        # wait here for monitors to be launched
        t = entk.Task()
        t.executable = 'sleep'
        t.arguments = ['10']
        s = entk.Stage()
        s.add_tasks(t)
        m.add_stages(s)
        s2 = entk.Stage()
        s2.add_tasks(self.generate_rp_monitor())
        s2.add_tasks(self.generate_hw_monitors())
        m.add_stages(s2)
        return m
    '''
    def run_workflow(self):
        # services are added
        #self.add_soma_service()
        # monitor pipeline
        #m = self.generate_monitor_pipeline()
        self.am.workflow = []
        for i in range(self.args.num_pipelines):
            p = self.generate_pipeline(i+1)
            self.am.workflow.append(p)
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    mvp.set_resource(res_desc = {
        'resource': 'ornl.summit_jsrun',
#        'queue'   : 'debug',
        'queue'   : mvp.args.queue,
#        'queue'   : 'default',
        'walltime': 60, #MIN
        'cpus'    : 42 * mvp.args.num_nodes,
        'gpus'    : 6 * mvp.args.num_nodes,
        'project' : mvp.args.project_id
        })
    mvp.run_workflow()
