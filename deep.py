from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
import tensorflow as tf
import numpy as np

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
MODEL_NAME="mancala"

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):
        self.model = self.create_model() # .fit every step
        self.target_model = self.create_model() # .predict every step 
        self.target_model.set_weights(self.model.get_weights()) 

        # take a random sample of the examples, as to not have batch_size=1 
        # for overfitting to individual samples
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(5,2)))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(4, activation="linear"))

        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        model.summary()
        return model
    
    def update_replay_memory(self, transition):
        # transition = observation space, action, reward, new observation space, done or not
        self.replay_memory.append(transition)

    def get_qs(self, state : np.array, step):
        # -1 on reshape means "figure out this dimension automatically"
        return self.model_predict(state.reshape(-1, *state.shape))
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)