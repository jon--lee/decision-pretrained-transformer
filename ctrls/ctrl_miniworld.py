import numpy as np
import scipy
from skimage.transform import resize
import torch
from torchvision.transforms import transforms

from ctrls.ctrl_bandit import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_shape = (25, 25, 3)


class MiniworldOptPolicy(Controller):
    def __init__(self, env, batch_size=1, save_video=False, filename_template=''):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.save_video = save_video
        self.filename_template = filename_template
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def reset(self):
        return

    def act(self, state, pose, angle):
        actions = []
        for i in range(self.batch_size):
            actions.append(self.env.envs[i].opt_a(state[i], pose[i], angle[i]))
        actions = np.array(actions)

        zeros = np.zeros((self.batch_size, self.env.action_space.n))
        zeros[np.arange(self.batch_size), actions] = 1
        return zeros


class MiniworldRandPolicy(MiniworldOptPolicy):
    def __init__(self, env, batch_size=1):
        super().__init__(env, batch_size=batch_size)

    def act(self, state, pose, angle):
        actions = np.random.randint(
            self.env.action_space.n, size=self.batch_size)
        zeros = np.zeros((self.batch_size, self.env.action_space.n))
        zeros[np.arange(self.batch_size), actions] = 1
        return zeros


class MiniworldTransformerController(Controller):
    def __init__(self, model, batch_size=1, sample=False, save_video=False, filename_template=''):
        self.model = model
        self.action_dim = 4
        self.horizon = model.horizon
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.sample = sample
        self.temp = 1.0
        self.batch_size = batch_size
        self.save_video = save_video
        self.filename_template = filename_template

    def act(self, image, pose, angle):
        images = np.array(image)
        if self.batch_size == 1:
            images = images[None, :]

        assert len(images.shape) == 4
        images = [resize(image, target_shape, anti_aliasing=True)
                  for image in images]
        images = [self.transform(image) for image in images]

        images = torch.stack(images).float().to(device)
        assert images.shape[1] == 3
        assert images.shape[2] == 25
        assert images.shape[3] == 25
        self.batch['query_images'] = images

        if self.batch_size == 1:
            pose = [pose]
            angle = [angle]

        self.batch['query_states'] = torch.tensor(
            np.array(angle)).float().to(device)

        actions = self.model(self.batch).cpu().detach().numpy()
        if self.batch_size == 1:
            actions = actions[0]

        if self.sample:
            if self.batch_size > 1:
                action_indices = []
                for idx in range(self.batch_size):
                    probs = scipy.special.softmax(actions[idx] / self.temp)
                    action_indices.append(np.random.choice(
                        np.arange(self.action_dim), p=probs))
            else:
                probs = scipy.special.softmax(actions / self.temp)
                action_indices = [np.random.choice(
                    np.arange(self.action_dim), p=probs)]
        else:
            action_indices = np.argmax(actions, axis=-1)

        actions = np.zeros((self.batch_size, self.action_dim))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        if self.batch_size == 1:
            actions = actions[0]
        return actions
