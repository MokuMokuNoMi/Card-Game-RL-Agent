import time
import torch.nn

import tensorboard
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
import optim_env

from experiment_grid import ExperimentGrid


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        if infos and "episode" in infos[0]:
            for info in infos:
                for item in info:
                    if item not in ["episode", "terminal_observation"]:
                        self.logger.record_mean(f"env/mean_{item}", info[item])
            self.logger.record("time/time_passed", time.time() - self.time)
        return True


def stable_baselines_run(
        log_dir=None,
        log_name=None,
        algorithm=PPO,
        time_steps=1_000_000,
        num_envs=4,
        model_kwargs=None,
        eval_policy=False,
):
    """
    Runs the environment with a model from the stable baseline library
    Args:
        data_suite (str): "Folder containing data csvs in Data directory"
        log_dir (str): Directory to save tensorboard logs and model to
        log_name (str): Name of tensorboard log
        algorithm: Stable Baselines 3 Algorithm to use
        time_steps (int): Number of time steps to train for
        num_envs (int): Number of parallel environments to run
        model_kwargs (dict): Arguments to be passed to algorithm
        eval_policy (bool): Whether to evaluate the policy after training
    """

    log_name = algorithm.__name__ if log_name is None else log_name
    log_name += f"_{num_envs}ENVS"
    if model_kwargs and "n_epochs" in model_kwargs:
        log_name += f'_{model_kwargs["n_epochs"]}EPOCHS'

    train_env = make_vec_env(
        env_id="optim-v0",
        n_envs=num_envs,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = make_vec_env(
        env_id="optim-v0",
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    model = algorithm("MultiInputPolicy", train_env, tensorboard_log=f"runs/{log_dir}", **model_kwargs)

    custom_log_cb = TensorboardCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"model_files/{log_dir}/BEST",
        eval_freq=int(16000),
        deterministic=False,
        verbose=0,
        render=True,
    )
    checkpoint_cb = CheckpointCallback(10000, f"model_files/{log_dir}", name_prefix=f"{log_name}")

    model.learn(
        total_timesteps=time_steps,
        tb_log_name=f"{log_name}",
        callback=[custom_log_cb, eval_cb, checkpoint_cb],
    )

    if eval_policy:
        mean, _ = evaluate_policy(model, env=eval_env, n_eval_episodes=4, render=False)
        print()
        print()
        print(f"Eval mean reward: {mean}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    eg = ExperimentGrid(stable_baselines_run)
    eg.add("algorithm", PPO)
    eg.add("model_kwargs:learning_rate", 8e-5)
    eg.add("model_kwargs:n_epochs", 10)
    # eg.add("model_kwargs:n_steps", 8)
    eg.add("model_kwargs:clip_range_vf", .5)
    eg.add("model_kwargs:verbose", 1)
    eg.add("model_kwargs:vf_coef", 1)
    eg.add("model_kwargs:ent_coef", 0)
    # eg.add("model_kwargs:target_kl", 0)
    # eg.add("model_kwargs:seed", 1)
    eg.add("model_kwargs:verbose", 1)
    eg.add("model_kwargs:policy_kwargs:activation_fn", torch.nn.LeakyReLU)
    eg.add("num_envs", [8])
    eg.add("time_steps", [1_000_000_000])
    eg.run()
