import tensorflow as tf
import numpy as np

input_size = 9
output_size = 2
layer_configurations = [15, 10, output_size]

class GameEnvironment:
    def __init__(self, model, input_size, output_size):
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.reset()

    def reset(self):
        self.state = self.initialize_game()
        return self.state

    def step(self, action):
        new_state, feedback, done = self.apply_action(action)
        reward = self.evaluate_reward(feedback)  # Convert feedback into reward using the new function
        if done:
            self.died()
        return new_state, reward, done

    def initialize_game(self):
        return np.random.rand(self.input_size)

    def apply_action(self, action):
        new_state = np.random.rand(self.input_size)
        feedback = np.random.choice([-3, -2, -1, 1, 2, 3])  # Random feedback for demonstration
        done = np.random.choice([True, False], p=[0.05, 0.95])
        return new_state, feedback, done

    def died(self):
        print("Agent has died. Resetting the game...")
        self.reset()

    def evaluate_reward(self, feedback):
        # Map the feedback levels to specific reward values
        reward_mapping = {
            -1: -0.5,  # Not good
            -2: -1,    # Bad
            -3: -1.5,  # VERY BAD
             1: 0.5,   # Better
             2: 1,     # Good
             3: 1.5    # VERY GOOD
        }
        return reward_mapping.get(feedback, 0)  # Return 0 if an unexpected feedback is given

def build_model(input_size, layer_configurations):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layer_configurations[0], activation='relu', input_shape=(input_size,)))
    for neurons in layer_configurations[1:]:
        model.add(tf.keras.layers.Dense(neurons, activation='relu' if neurons != layer_configurations[-1] else 'sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model(input_size, layer_configurations)
env = GameEnvironment(model, input_size, output_size)

def test_model(env, tests):
    for _ in range(tests):
        input_vector = np.random.rand(input_size)
        action = model.predict(np.array([input_vector]))[0]
        state, reward, done = env.step(action)
        print(f"Input: {input_vector}, Action: {action}, Reward: {reward}, Done: {done}")

# Run tests with the environment
test_model(env, 3)
# Simulate agent dying and reset the environment
env.died()
# Test the model again after reset
test_model(env, 3)
