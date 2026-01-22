from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from collections import deque
import time
import random
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 2000
MINIBATCH_SIZE = 32
DISCOUNT_RATE = 0.98
UPDATE_TARGET_EVERY = 1000
MODEL_NAME="mancala"


class DQNAgent:
    def __init__(self, name="DQNAgent"):
        self.name = name
        
        self.model = self.create_model() # .fit every step
        self.target_model = self.create_model() # .predict every step, we want consistency in predictions
        self.target_model.set_weights(self.model.get_weights()) 

        # take a random sample of the examples, as to not have batch_size=1 
        # and overfit gradient to individual samples
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.train_step_counter = 0
        self.target_update_counter = 0

        self.writer = tf.summary.create_file_writer(f"logs/{MODEL_NAME}-{name}")
        self.global_step = 0

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(14,)))
        # Normalization layer necessary?
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(6, activation="linear"))

        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        model.summary()
        return model
    
    def update_replay_memory(self, transition):
        # transition = observation space, action, reward, new observation space, done or not
        self.replay_memory.append(transition)

    def get_qs(self, state : np.array):
        # -1 on reshape means "figure out this dimension automatically"
        return self.model(state.reshape(-1, *state.shape), verbose=0)[0]
    
    def train(self, terminal_state, train_every=4):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # skip some training steps
        self.train_step_counter += 1
        self.target_update_counter += 1

        if self.train_step_counter % train_every != 0:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model(current_states, verbose=0)

        # target model is updated less often to prevent instability (prevent from chasing its own moving predictions).
        new_current_states = np.array([transition[3] for transition in minibatch]) # new obs space
        future_qs_list = self.target_model(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT_RATE * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index].numpy()

            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        history = self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

        # Manually log metrics to TensorBoard
        self.global_step += 1
        with self.writer.as_default():
            tf.summary.scalar('loss', history.history['loss'][0], step=self.global_step)
            tf.summary.scalar('accuracy', history.history['accuracy'][0], step=self.global_step)
            self.writer.flush()

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            print(f"Now setting target weights for agent: {self.name}")
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0