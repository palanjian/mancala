from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from collections import deque
import time
import random
import tensorflow as tf
import numpy as np
from dqntboard import ModifiedTensorBoard

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 64
DISCOUNT_RATE = 0.99
UPDATE_TARGET_EVERY = 5
MODEL_NAME="mancala"


class DQNAgent:
    def __init__(self):
        self.model = self.create_model() # .fit every step
        self.target_model = self.create_model() # .predict every step, we want consistency in predictions
        self.target_model.set_weights(self.model.get_weights()) 

        # take a random sample of the examples, as to not have batch_size=1 
        # and overfit gradient to individual samples
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(5,2)))
        # Normalization layer necessary?
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
        return self.model_predict(state.reshape(-1, *state.shape))[0]
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # target model is updated less often to prevent instability (prevent from chasing its own moving predictions).
        new_current_states = np.array([transition[3] for transition in minibatch]) # new obs space
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT_RATE * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None
        )

        # used to determine if we want to update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0