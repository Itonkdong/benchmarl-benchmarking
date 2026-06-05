import torch
from benchmarl.algorithms import MappoConfig, QmixConfig, MasacConfig, MaddpgConfig, VdnConfig, IppoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, LstmConfig, CnnConfig
from pprint import pprint

def main():
    exp_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.BALANCE.get_from_yaml()
    alg = MasacConfig.get_from_yaml()
    model = MlpConfig.get_from_yaml()
    critic_model = MlpConfig.get_from_yaml()


    # vars(test_env.scenario) - How to get these params
    # ==================================================

    # task.config["max_steps"] = 400
    # task.config["package_mass"] = 5


    # task.config["collision_reward"] = -0.015
    # task.config["use_vel_controller"] = True
    # task.config["max_speed_1"] = 0.25
    # task.config["observe_joint_angle"] = True
    # task.config["pos_shaping_factor"] = 3
    # task.config["rot_shaping_factor"] = 1.03
    # task.config
    # task.config["energy_reward_coeff"] = -0.01
    # task.config["use_vel_controller"] = True
    # task.config["max_speed_1"] = 0.7
    # task.config["observe_joint_angle"] = True

    # alg.entropy_coef = 0.05
    # exp_config.lr = 0.00002
    # alg.critic_coef = 1.5
    # model.num_cells = [256, 256, 128]
    # ==================================================


    ITERATIONS = 100
    FRAMES_PER_BATCH = exp_config.on_policy_collected_frames_per_batch
    DO_EVERY = 20

    exp_config.train_device = "mps"

    # exp_config.checkpoint_interval = FRAMES_PER_BATCH * DO_EVERY
    # exp_config.on_policy_n_envs_per_worker = 16
    # exp_config.checkpoint_at_end = True
    # alg.share_param_critic = False

    exp_config.max_n_iters = ITERATIONS
    exp_config.max_n_frames = None
    exp_config.evaluation_interval = FRAMES_PER_BATCH * DO_EVERY

    # exp_config.on_policy_collected_frames_per_batch = 8192
    # exp_config.on_policy_n_envs_per_worker = 16
    exp_config.on_policy_minibatch_size = 300
    # exp_config.on_policy_n_minibatch_iters = 20
    # ==================================================

    experiment = Experiment(
        task=task,
        algorithm_config=alg,
        model_config=model,
        critic_model_config=critic_model,
        seed=0,
        config=exp_config
    )
    experiment.run()


if __name__ == '__main__':
    main()