import json #JavaScript Object Notation, used for storing dictionaries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd

#BOTH ENVIRONMENT AND AGENT MODELLED IN SAME CLASS (AS DONE IN GYM LIB)
class Catch(object):
	def __init__(self, grid_size=10):
		self.grid_size = grid_size
		self.reset()

	def _update_state(self, action):
		"""
		Input: action and states
		Ouput: new states and reward
		"""
		state = self.state #THIS self.state IS DEFINED BELOW
		if action == 0:  # 0 means LEFT
			action = -1 #LEFT means moving -1 on x-axis
		elif action == 1:  # 1 means STAY
			action = 0 #STAY means moving 0 on x-axis
		else: #2 means RIGHT
			action = 1 #RIGHT means moving 1 on x-axis
		f0, f1, basket = state[0]
		new_basket = min(max(1, basket + action), self.grid_size-1)
		f0 += 1
		out = np.asarray([f0, f1, new_basket]) #[f0, f1, new_basket] shape is (3,)
		out = out[np.newaxis] #[[f0, f1, new_basket]] shape = (1,3)

		assert len(out.shape) == 2
		self.state = out

	def _draw_state(self):
		im_size = (self.grid_size,)*2 #im_size = (grid_size,grid_size)
		state = self.state[0] #self.state[0] = [f0,f1,basket]
		canvas = np.zeros(im_size) #matrix/2D array of shape im_size
		canvas[state[0], state[1]] = 1  #draw fruit
        #canvas takes coordinates as (y,x)
		canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        #if basket position is 2, basket is drawn from 1 to 3 ie length = 3
		return canvas

	def _get_reward(self):
		fruit_row, fruit_col, basket = self.state[0] #[f0,f1,basket]
		if fruit_row == self.grid_size-1: #if fruit has reached bottom
			if abs(fruit_col - basket) <= 1: #check if fruit/basket overlap
				return 1 #if yes give reward
			else:
				return -1 #if they don't overlap give punishment
		else:
			return 0 #if fruit has not reached bottom yet, don't give reward

	def _is_over(self):
		if self.state[0, 0] == self.grid_size-1: #if f0 == grid_size-1
			return True
		else:
			return False

	def observe(self):
		canvas = self._draw_state()
		return canvas.reshape((1, -1)) #FLATTEN OPERATION WITH 1 ROW

	def act(self, action):
		#print(self.state) 
        #if you want to print state before action, uncomment the above line
		self._update_state(action)
		reward = self._get_reward()
		game_over = self._is_over()
		return self.observe(), reward, game_over

	def reset(self):
		n = np.random.randint(0, self.grid_size-1, size=1)
		m = np.random.randint(1, self.grid_size-2, size=1)
		self.state = np.asarray([0, n, m])[np.newaxis] 
        #converts [0,n,m] to array and np.newaxis makes it [[0,n,m]]


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory #no. of tuples it can store
        self.memory = list() #empty list []
        self.discount = discount #Discount Factor Gamma in Q-Learning

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        #states = [state_t,action_t,reward_t,state_t+1] ie s,a,r,s'
        #state_t = [[f0,f1,basket]] at time t
        #state_t+1 = [[f0,f1,basket]] at time t+1
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t #same as inputs[i] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 1000 #no of fruits
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    grid_size = 10

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu')) #Rectified Linear Unit Activation Function
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse") #sgd is stochastic gradient descent mse is mean squared error

    # If you want to continue training from a previous model, just uncomment the line below
    # model.load_weights("model.h5")

    # Define environment/game
    env = Catch(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)