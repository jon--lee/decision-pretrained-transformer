import imageio
import numpy as np
from skimage.transform import resize
import torch

from envs.darkroom_env import DarkroomEnvVec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_shape = (25, 25, 3)


class MiniworldEnvVec(DarkroomEnvVec):
    """
    Vectorized environment for MiniWorld.
    """

    def __init__(self, envs):
        super().__init__(envs)
        self.action_space = envs[0].action_space

    def reset(self):
        return [env.reset()[0] for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, _, {}

    def opt_a(self, x):
        return [env.opt_a(x) for env in self._envs]

    def deploy(self, ctrl):
        images = self.reset()
        pose = [env.agent.pos[[0, -1]] for env in self._envs]
        angle = [env.agent.dir_vec[[0, -1]] for env in self._envs]

        obs = []
        states = []
        acts = []
        next_obs = []
        rews = []
        done = False

        if ctrl.save_video:
            videos = [[] for _ in range(self.num_envs)]

        while not done:

            action = ctrl.act(images, pose, angle)

            images = [resize(image, target_shape, anti_aliasing=True)
                      for image in images]
            image_tensor = torch.stack(
                [ctrl.transform(image) for image in images])
            obs.append(image_tensor)
            states.append(angle)
            acts.append(action)

            images, rew, done, _, _ = self.step(np.argmax(action, axis=-1))
            pose = [env.agent.pos[[0, -1]] for env in self._envs]   # unused
            angle = [env.agent.dir_vec[[0, -1]] for env in self._envs]
            done = all(done)

            rews.append(rew)
            next_image_tensor = torch.stack(
                [ctrl.transform(image) for image in images])
            next_obs.append(next_image_tensor)

            if ctrl.save_video:
                imgs = [
                    env.unwrapped.render(goal_text=True, action=ac)
                    for env, ac in zip(self._envs, np.argmax(action, axis=-1))]
                for i, img in enumerate(imgs):
                    videos[i].append(img)

        if ctrl.save_video:
            videos = np.array(videos)
            for i in range(self.num_envs):
                imageio.mimsave(ctrl.filename_template(env_id=i), videos[i])

        return (
            torch.stack(obs, axis=1),
            np.stack(states, axis=1),
            np.stack(acts, axis=1),
            torch.stack(next_obs, axis=1),
            np.stack(rews, axis=1),
        )
