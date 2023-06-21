import os
from collections import deque
from env.aquarium import Aquarium
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

print("TensorFlow version: " + tf.__version__)
print("Keras version: " + keras.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.config.get_visible_devices())

env = Aquarium(render_mode="None")

batch_size = 32

# A2C, PPO spróbuj zaimplementować


def short_model():
    surr_input = keras.layers.Input(
        shape=(env.angle_precision, 5), name="surr_input")
    surr_input = tf.expand_dims(surr_input, -1)
    surr_conv_1 = keras.layers.Conv1D(
        30, 3, activation="relu", name="surr_conv_1", input_shape=(env.angle_precision,))(surr_input)
    surr_maxpool_1 = keras.layers.MaxPool1D(name="surr_maxpool_1")(surr_conv_1)
    surr_conv_2 = keras.layers.Conv1D(
        20, 3, activation="relu", name="surr_conv_2")(surr_maxpool_1)
    surr_maxpool_2 = keras.layers.MaxPool1D(name="surr_maxpool_2")(surr_conv_2)
    surr_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="surr_dense_1")(surr_maxpool_2)
    surr_flatten = keras.layers.Flatten()(surr_dense_1)
    surr_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="surr_dense_2")(surr_flatten)

    data_input = keras.layers.Input(
        shape=(env.angle_precision,), name="data_input")
    data_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="data_dense_1")(data_input)
    data_dense_2 = keras.layers.Dense(round(
        env.angle_precision / 4), activation="relu", name="data_dense_2")(data_dense_1)

    concat = keras.layers.Concatenate(
        name="concat")([surr_dense_2, data_dense_2])

    out_1 = keras.layers.Dense(128, activation="relu", name="out_1")(concat)
    out_2 = keras.layers.Dense(64, activation="relu", name="out_2")(out_1)
    outputs = keras.layers.Dense(
        10, activation="linear", name="outputs")(out_2)

    model = tf.keras.Model(inputs=[surr_input, data_input], outputs=outputs)
    return model


def new_model():
    aquarium_input = keras.layers.Input(
        shape=(env.angle_precision,), name="aquarium_input")
    aquarium_input = tf.expand_dims(aquarium_input, -1)
    aquarium_conv_1 = keras.layers.Conv1D(
        20, 3, activation="relu", name="aquarium_conv_1", input_shape=(env.angle_precision,))(aquarium_input)
    aquarium_maxpool_1 = keras.layers.MaxPool1D(
        name="aquarium_maxpool_1")(aquarium_conv_1)
    aquarium_conv_2 = keras.layers.Conv1D(
        10, 3, activation="relu", name="aquarium_conv_2")(aquarium_maxpool_1)
    aquarium_maxpool_2 = keras.layers.MaxPool1D(
        name="aquarium_maxpool_2")(aquarium_conv_2)
    aquarium_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="aquarium_dense_1")(aquarium_maxpool_2)
    aquarium_flatten = keras.layers.Flatten()(aquarium_dense_1)
    aquarium_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="aquarium_dense_2")(aquarium_flatten)

    walls_input = keras.layers.Input(
        shape=(env.angle_precision,), name="walls_input")
    walls_input = tf.expand_dims(walls_input, -1)
    walls_conv_1 = keras.layers.Conv1D(
        20, 3, activation="relu", name="walls_conv_1", input_shape=(env.angle_precision,))(walls_input)
    walls_maxpool_1 = keras.layers.MaxPool1D(
        name="walls_maxpool_1")(walls_conv_1)
    walls_conv_2 = keras.layers.Conv1D(
        10, 3, activation="relu", name="walls_conv_2")(walls_maxpool_1)
    walls_maxpool_2 = keras.layers.MaxPool1D(
        name="walls_maxpool_2")(walls_conv_2)
    walls_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="walls_dense_1")(walls_maxpool_2)
    walls_flatten = keras.layers.Flatten()(walls_dense_1)
    walls_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="walls_dense_2")(walls_flatten)

    food_input = keras.layers.Input(
        shape=(env.angle_precision,), name="food_input")
    food_input = tf.expand_dims(food_input, -1)
    food_conv_1 = keras.layers.Conv1D(
        20, 3, activation="relu", name="food_conv_1", input_shape=(env.angle_precision,))(food_input)
    food_maxpool_1 = keras.layers.MaxPool1D(name="food_maxpool_1")(food_conv_1)
    food_conv_2 = keras.layers.Conv1D(
        10, 3, activation="relu", name="food_conv_2")(food_maxpool_1)
    food_maxpool_2 = keras.layers.MaxPool1D(name="food_maxpool_2")(food_conv_2)
    food_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="food_dense_1")(food_maxpool_2)
    food_flatten = keras.layers.Flatten()(food_dense_1)
    food_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="food_dense_2")(food_flatten)

    fishes_input = keras.layers.Input(
        shape=(env.angle_precision,), name="fishes_input")
    fishes_input = tf.expand_dims(fishes_input, -1)
    fishes_conv_1 = keras.layers.Conv1D(
        20, 3, activation="relu", name="fishes_conv_1", input_shape=(env.angle_precision,))(fishes_input)
    fishes_maxpool_1 = keras.layers.MaxPool1D(
        name="fishes_maxpool_1")(fishes_conv_1)
    fishes_conv_2 = keras.layers.Conv1D(
        10, 3, activation="relu", name="fishes_conv_2")(fishes_maxpool_1)
    fishes_maxpool_2 = keras.layers.MaxPool1D(
        name="fishes_maxpool_2")(fishes_conv_2)
    fishes_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="fishes_dense_1")(fishes_maxpool_2)
    fishes_flatten = keras.layers.Flatten()(fishes_dense_1)
    fishes_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="fishes_dense_2")(fishes_flatten)

    sharks_input = keras.layers.Input(
        shape=(env.angle_precision,), name="sharks_input")
    sharks_input = tf.expand_dims(sharks_input, -1)
    sharks_conv_1 = keras.layers.Conv1D(
        20, 3, activation="relu", name="sharks_conv_1", input_shape=(env.angle_precision,))(sharks_input)
    sharks_maxpool_1 = keras.layers.MaxPool1D(
        name="sharks_maxpool_1")(sharks_conv_1)
    sharks_conv_2 = keras.layers.Conv1D(
        10, 3, activation="relu", name="sharks_conv_2")(sharks_maxpool_1)
    sharks_maxpool_2 = keras.layers.MaxPool1D(
        name="sharks_maxpool_2")(sharks_conv_2)
    sharks_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="sharks_dense_1")(sharks_maxpool_2)
    sharks_flatten = keras.layers.Flatten()(sharks_dense_1)
    sharks_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="sharks_dense_2")(sharks_flatten)

    data_input = keras.layers.Input(
        shape=(env.angle_precision,), name="data_input")
    data_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="data_dense_1")(data_input)
    data_dense_2 = keras.layers.Dense(round(
        env.angle_precision / 4), activation="relu", name="data_dense_2")(data_dense_1)

    concat = keras.layers.Concatenate(
        name="concat")([aquarium_dense_2, walls_dense_2, food_dense_2, fishes_dense_2, sharks_dense_2, data_dense_2])

    out_1 = keras.layers.Dense(128, activation="relu", name="out_1")(concat)
    out_2 = keras.layers.Dense(64, activation="relu", name="out_2")(out_1)
    outputs = keras.layers.Dense(
        env.action_amount, activation="linear", name="outputs")(out_2)

    model = tf.keras.Model(inputs=[aquarium_input, walls_input, food_input,
                           fishes_input, sharks_input, data_input], outputs=outputs)
    return model


def create_model():
    surr_input = keras.layers.Input(
        shape=(env.angle_precision,), name="surr_input")
    surr_input = tf.expand_dims(surr_input, -1)
    surr_conv_1 = keras.layers.Conv1D(
        30, 3, activation="relu", name="surr_conv_1", input_shape=(env.angle_precision,))(surr_input)
    surr_maxpool_1 = keras.layers.MaxPool1D(name="surr_maxpool_1")(surr_conv_1)
    surr_conv_2 = keras.layers.Conv1D(
        20, 3, activation="relu", name="surr_conv_2")(surr_maxpool_1)
    surr_maxpool_2 = keras.layers.MaxPool1D(name="surr_maxpool_2")(surr_conv_2)
    surr_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="surr_dense_1")(surr_maxpool_2)
    surr_flatten = keras.layers.Flatten()(surr_dense_1)
    surr_dense_2 = keras.layers.Dense(
        env.angle_precision / 4, activation="relu", name="surr_dense_2")(surr_flatten)
    # surr_dense_1 = keras.layers.Dense(
    #     env.angle_precision * 2, activation="relu", name="surr_dense_1", input_shape=(120,))(surr_input)
    # surr_dense_2 = keras.layers.Dense(
    #     env.angle_precision, activation="relu", name="surr_dense_2")(surr_dense_1)
    # surr_dense_3 = keras.layers.Dense(round(
    #     env.angle_precision / 2), activation="relu", name="surr_dense_3")(surr_dense_2)

    data_input = keras.layers.Input(
        shape=(env.angle_precision,), name="data_input")
    data_dense_1 = keras.layers.Dense(
        env.angle_precision / 2, activation="relu", name="data_dense_1")(data_input)
    data_dense_2 = keras.layers.Dense(round(
        env.angle_precision / 4), activation="relu", name="data_dense_2")(data_dense_1)

    concat = keras.layers.Concatenate(
        name="concat")([surr_dense_2, data_dense_2])

    out_1 = keras.layers.Dense(128, activation="relu", name="out_1")(concat)
    out_2 = keras.layers.Dense(64, activation="relu", name="out_2")(out_1)
    outputs = keras.layers.Dense(
        10, activation="linear", name="outputs")(out_2)

    model = tf.keras.Model(inputs=[surr_input, data_input], outputs=outputs)
    return model


def epsilon_greedy_policy(state, model, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 10)
    else:
        # surr = np.array(state["observation"]["surrounding"]["conjoined"])
        # surr = np.reshape(surr, (1, env.angle_precision))
        aqu = np.array(state["observation"]["surrounding"]["aquarium"])
        aqu = np.reshape(aqu, (1, env.angle_precision))
        walls = np.array(state["observation"]["surrounding"]["walls"])
        walls = np.reshape(walls, (1, env.angle_precision))
        food = np.array(state["observation"]["surrounding"]["food"])
        food = np.reshape(food, (1, env.angle_precision))
        fishes = np.array(state["observation"]["surrounding"]["fishes"])
        fishes = np.reshape(fishes, (1, env.angle_precision))
        sharks = np.array(state["observation"]["surrounding"]["sharks"])
        sharks = np.reshape(sharks, (1, env.angle_precision))
        data = np.array(state["observation"]["data"])
        data = np.reshape(data, (1, env.angle_precision))
        Q_values = model.predict(
            [[aqu], [walls], [food], [fishes], [sharks], [data]])
        return np.argmax(Q_values[0])


fish_model = new_model()
shark_model = new_model()

models = {
    "fish": fish_model,
    "shark": shark_model
}

fish_replay_buffer = deque(maxlen=20000)
shark_replay_buffer = deque(maxlen=20000)

buffers = {
    "fish": fish_replay_buffer,
    "shark": shark_replay_buffer
}


def sample_experiences(batch_size, replay_buffer):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


batch_size = 32
discount_factor = 0.99
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
n_outputs = env.action_amount

work_folder = 'C:/ml/models/aquarium/conv14/'


def training_step(batch_size, model, replay_buffer, episode):
    filename = work_folder + str(episode) + '/'
    experiences = sample_experiences(batch_size, replay_buffer=replay_buffer)
    states, actions, rewards, next_states, dones = experiences
    # print(states)
    s_a = []
    s_b = []
    s_w = []
    s_fo = []
    s_fi = []
    s_s = []
    ns_a = []
    ns_b = []
    ns_w = []
    ns_fo = []
    ns_fi = []
    ns_s = []
    for s in states:
        s_a.append(s["observation"]["surrounding"]["aquarium"])
        s_w.append(s["observation"]["surrounding"]["walls"])
        s_fo.append(s["observation"]["surrounding"]["food"])
        s_fi.append(s["observation"]["surrounding"]["fishes"])
        s_s.append(s["observation"]["surrounding"]["sharks"])
        s_b.append(s["observation"]["data"])
    for n in next_states:
        ns_a.append(s["observation"]["surrounding"]["aquarium"])
        ns_w.append(s["observation"]["surrounding"]["walls"])
        ns_fo.append(s["observation"]["surrounding"]["food"])
        ns_fi.append(s["observation"]["surrounding"]["fishes"])
        ns_s.append(s["observation"]["surrounding"]["sharks"])
        ns_b.append(n["observation"]["data"])
    s_a = np.asarray(s_a)
    s_b = np.asarray(s_b)
    s_w = np.asarray(s_w)
    s_fo = np.asarray(s_fo)
    s_fi = np.asarray(s_fi)
    s_s = np.asarray(s_s)
    ns_a = np.asarray(ns_a)
    ns_b = np.asarray(ns_b)
    ns_w = np.asarray(ns_w)
    ns_fo = np.asarray(ns_fo)
    ns_fi = np.asarray(ns_fi)
    ns_s = np.asarray(ns_s)
    # with open(filename + 's_a.txt', 'a') as f:
    #     f.write(str(s_a))
    # with open(filename + 's_b.txt', 'a') as f:
    #     f.write(str(s_b))
    # with open(filename + 'ns_a.txt', 'a') as f:
    #     f.write(str(ns_a))
    # with open(filename + 's_b.txt', 'a') as f:
    #     f.write(str(ns_b))
    next_Q_values = model.predict(
        [ns_a, ns_w, ns_fo, ns_fi, ns_s, ns_b], batch_size=batch_size)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    # with open(filename + 'mnQv.txt', 'a') as f:
    #     f.write(str(max_next_Q_values))
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * max_next_Q_values)
    # with open(filename + 'tQv.txt', 'a') as f:
    #     f.write(str(target_Q_values))
    mask = tf.one_hot(actions, n_outputs)
    # with open(filename + 'one_hot.txt', 'a') as f:
    #     f.write(str(mask))
    with tf.GradientTape() as tape:
        all_Q_values = model([s_a, s_w, s_fo, s_fi, s_s, s_b])
        # with open(filename + 'aQv.txt', 'a') as f:
        #     f.write(str(all_Q_values))
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        # with open(filename + 'Qv.txt', 'a') as f:
        #     f.write(str(Q_values))
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        # with open(filename + 'loss.txt', 'a') as f:
        #     f.write(str(loss))
    grads = tape.gradient(loss, model.trainable_variables)
    # with open(filename + 'grads.txt', 'a') as f:
    #     f.write(str(grads))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # with open(filename + 'apply_grad.txt', 'a') as f:
    #     f.write("apply_grad")


agent_rewards = {}
# [name][episode]: []
howManyEpisodes = 2000
train_start = howManyEpisodes / 10
if train_start > 100:
    train_start = 100
for episode in range(howManyEpisodes):
    env.reset()
    dir = work_folder + str(episode)
    if not os.path.exists(dir):
        os.makedirs(dir)
    truncated = []
    x = 1 - episode / howManyEpisodes
    if x < 0:
        x = 0
    epsilon = max(x, 0.01)
    for agent in env.agent_iter(env.max_timesteps):
        if agent not in agent_rewards:
            agent_rewards[agent] = {e: [] for e in range(howManyEpisodes)}
        if agent not in truncated:
            observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            truncated.append(agent)
        agentType = "fish"
        if "shark" in agent:
            agentType = "shark"
        if agent in truncated:
            action = None
            env.step(action)
        else:
            action = epsilon_greedy_policy(
                observation, models[agentType], epsilon)
            env.step(action)
            next_observation, reward, termination, truncation, info = env.get_agent_observation(
                agentName=agent)
            buffers[agentType].append(
                (observation, action, reward, next_observation, truncation))
            agent_rewards[agent][episode].append(reward)

    if episode > train_start:
        training_step(batch_size, models["fish"], buffers["fish"], episode)
        training_step(batch_size, models["shark"], buffers["shark"], episode)
        if episode % 200 == 0:
            models["fish"].save(
                work_folder + 'fish.h5')
            models["shark"].save(
                work_folder + 'shark.h5')
            with open(work_folder + 'agent_rewards.json', 'w') as outfile:
                json.dump(agent_rewards, outfile)
env.close()

if episode % 200 != 0:
    models["fish"].save(work_folder + 'fish.h5')
    models["shark"].save(work_folder + 'shark.h5')
    with open(work_folder + 'agent_rewards.json', 'w') as outfile:
        json.dump(agent_rewards, outfile)
# Directly from dictionary


# with open('C:/ml/models/aquarium/agent_rewards.json', 'w') as outfile:
#     json.dump(buffers, outfile)
