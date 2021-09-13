import torch
import pickle
import os
import shutil
from vae.dataman import DataManager


torch.set_num_threads(12)
DEFAULT_DEVICE = torch.device('cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
DEFAULT_DEVICE = torch.device('cuda:0')  # uncomment to use cuda cores...


class TrainingState:
    """
    Represents a picture of all states during training given by the
    Optimizer, Scheduler, and the Model.
    :param factory: function to return a triple with a initialized model, the optimizer and the scheduler
    """

    def __init__(self, factory, **kwargs):
        self.model, self.optimizer, self.scheduler = factory(**kwargs)
        self.history = []
        self.lrs = []

    def train_epoch(self, data: DataManager, batch_size: int):
        batches = 0
        accum = None
        for batch in data.get_batches(batch_size):  # get data shuffled in batches
            self.optimizer.zero_grad()
            summary = self.model.train_batch(batch)
            accum = summary if accum is None else [x + y for x, y in zip(accum, summary)]
            batches += 1
            self.optimizer.step()

        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()
        self.history.append([x / batches for x in accum])
        return accum[0] / batches  # loss evaluation average (assuming loss is 0 index in summary)

    def load(self, state_path):
        """
        Loads the state of a training process from a specific folder.
        :param state_path: the path containing the state binary files
        """
        self.model.load_state_dict(torch.load(state_path + "/model.pt"))
        self.optimizer.load_state_dict(torch.load(state_path + "/optimizer.pt"))
        with open(state_path + "/summary.bin", 'rb') as fp:
            self.history = pickle.load(fp)
        with open(state_path + "/lr.bin", 'rb') as fp:
            self.lrs = pickle.load(fp)

    def save(self, state_path):
        """
        Saves the current state of the training process at specific folder.
        Notice: use a different folder for each state.
        :param state_path: the path will contain the state binary files
        """
        os.makedirs(state_path, exist_ok=True)
        torch.save(self.model.state_dict(), state_path + "/model.pt")
        torch.save(self.optimizer.state_dict(), state_path + "/optimizer.pt")
        with open(state_path + "/summary.bin", 'wb') as fp:
            pickle.dump(self.history, fp)
        with open(state_path + "/lr.bin", 'wb') as fp:
            pickle.dump(self.lrs, fp)


def clear_training(states_path):
    """
    Delete all model states in the specific path
    """
    if os.path.exists(states_path):
        shutil.rmtree(states_path)


def get_last_model(states_path, factory, top_epoch=None, **kwargs):
    kwargs['epochs'] = 100
    kwargs['batch_size'] = 1024

    state = TrainingState(factory, **kwargs)
    state.model.to(torch.device('cpu'))

    state_id = 0
    while os.path.exists(states_path + "/state" + str(state_id + 1)) and (top_epoch is None or state_id*10 < top_epoch):
        state_id += 1

    if os.path.exists(states_path + "/state" + str(state_id)):
        state.load(states_path + "/state" + str(state_id))
        print('Loading previous state... ' + states_path + "/state" + str(state_id))
    else:
        print('Model not found!')
    return state.model


def start_training(states_path, data: DataManager, factory, batch_size=1024, epochs=100, **kwargs):
    kwargs['epochs'] = epochs
    kwargs['batch_size'] = batch_size

    state = TrainingState(factory, **kwargs)
    state.model.to(DEFAULT_DEVICE)
    data.set_device(DEFAULT_DEVICE)
    print("Using device: "+str(DEFAULT_DEVICE))

    state_id = 0
    while os.path.exists(states_path + "/state" + str(state_id + 1)):
        state_id += 1

    if os.path.exists(states_path + "/state" + str(state_id)):
        state.load(states_path + "/state" + str(state_id))
        print('Loading previous state... ' + states_path + "/state" + str(state_id))

    state.model.train()

    start_epoch = len(state.history)
    for epoch in range(start_epoch, epochs):
        state.train_epoch(data, batch_size)
        if (epoch + 1) % 10 == 0:
            print('Epoch ' + str(epoch) + ' \t Loss & info: '+str(state.history[-1]))
            state_id += 1
            state.save(states_path + "/state" + str(state_id))

    return state
