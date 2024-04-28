import tensorflow as tf
import numpy as np

# Define neural network architecture variables
input_size = 9
output_size = 2
layer_configurations = [15, 10, output_size]  # Number of neurons in each layer

class GameEnvironment:
    def __init__(self, model, input_size, output_size):
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.reset()

    def reset(self):
        # Reset the game to the initial state
        self.state = self.initialize_game()
        return self.state

    def step(self, action):
        # Apply the action in the game, receive new state, reward, and check if game is done
        new_state, reward, done = self.apply_action(action)
        if done:
            self.died()
        return new_state, reward, done

    def initialize_game(self):
        # Initialize the game state with random values based on input size
        return np.random.rand(self.input_size)

    def apply_action(self, action):
        # Random new state generation for simulation
        new_state = np.random.rand(self.input_size)
        # Calculate reward by comparing the first 'output_size' elements of the new state to the action
        reward = -np.sum(np.abs(new_state[:self.output_size] - action))  # Assume action directly compares to part of the state
        done = np.random.choice([True, False], p=[0.05, 0.95])  # 5% chance the game ends
        return new_state, reward, done

    def died(self):
        # Print message and reset game if the agent dies
        print("Agent has died. Resetting the game...")
        self.reset()

def build_model(input_size, layer_configurations):
    # Build a model dynamically based on the layer configurations
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layer_configurations[0], activation='relu', input_shape=(input_size,)))
    for neurons in layer_configurations[1:]:
        model.add(tf.keras.layers.Dense(neurons, activation='relu' if neurons != layer_configurations[-1] else 'sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize the model with specified configuration
model = build_model(input_size, layer_configurations)
env = GameEnvironment(model, input_size, output_size)

def test_model(env, tests):
    for _ in range(tests):
        input_vector = np.random.rand(input_size)  # Generate random inputs
        action = model.predict(np.array([input_vector]))[0]  # Model generates action based on input
        state, reward, done = env.step(action)  # Environment responds to the action
        print(f"Input: {input_vector}, Action: {action}, Reward: {reward}, Done: {done}")

# Run tests with the environment
test_model(env, 3)
# Simulate agent dying and reset the environment
env.died()
# Test the model again after reset
test_model(env, 3)
