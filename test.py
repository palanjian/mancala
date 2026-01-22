from keras.models import load_model
import numpy as np

from mancala import Mancala

model_name = 'best.keras'

model = load_model(model_name)
model.summary()

terminated = False
mancala_game = Mancala()
current_player = 2

while not terminated:
    print(mancala_game)

    if current_player == 1:
        action = int(input("Select a move! (1-6)"))
    
    else:
        state = mancala_game.get_obs_for(2)
        mask = np.array([not mancala_game.is_illegal(current_player, i) for i in range(0, 6)], dtype=np.int8)
        qs = model(state.reshape(-1, *state.shape), verbose=0)[0]
        masked_qs = np.where(mask, qs, -np.inf)
        action = np.argmax(masked_qs)

    print(f"Player {current_player} takes action {action}, resulting in state: ")
    legal, next_player, terminated, won, _ = mancala_game.move(current_player, action)
    current_player = next_player