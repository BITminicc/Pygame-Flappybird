import os
import sys
import random
import pygame
import argparse
import itertools
import torch
import numpy as np
import skimage.color
import torch.nn as nn
import skimage.exposure
import skimage.transform
import pickle
from collections import deque

# FPS
FPS = 30
# the screen size
SCREENWIDTH = 288
SCREENHEIGHT = 512
# the gap between pipes
PIPE_GAP_SIZE = 100
# the game image paths
NUMBER_IMAGE_PATHS = {
						'0': os.path.join(os.getcwd(), 'resources/images/0.png'),
						'1': os.path.join(os.getcwd(), 'resources/images/1.png'),
						'2': os.path.join(os.getcwd(), 'resources/images/2.png'),
						'3': os.path.join(os.getcwd(), 'resources/images/3.png'),
						'4': os.path.join(os.getcwd(), 'resources/images/4.png'),
						'5': os.path.join(os.getcwd(), 'resources/images/5.png'),
						'6': os.path.join(os.getcwd(), 'resources/images/6.png'),
						'7': os.path.join(os.getcwd(), 'resources/images/7.png'),
						'8': os.path.join(os.getcwd(), 'resources/images/8.png'),
						'9': os.path.join(os.getcwd(), 'resources/images/9.png')
					}
BIRD_IMAGE_PATHS = {
						'red': {'up': os.path.join(os.getcwd(), 'resources/images/redbird-upflap.png'),
								'mid': os.path.join(os.getcwd(), 'resources/images/redbird-midflap.png'),
								'down': os.path.join(os.getcwd(), 'resources/images/redbird-downflap.png')},
						'blue': {'up': os.path.join(os.getcwd(), 'resources/images/bluebird-upflap.png'),
								 'mid': os.path.join(os.getcwd(), 'resources/images/bluebird-midflap.png'),
								 'down': os.path.join(os.getcwd(), 'resources/images/bluebird-downflap.png')},
						'yellow': {'up': os.path.join(os.getcwd(), 'resources/images/yellowbird-upflap.png'),
								   'mid': os.path.join(os.getcwd(), 'resources/images/yellowbird-midflap.png'),
								   'down': os.path.join(os.getcwd(), 'resources/images/yellowbird-downflap.png')}
					}
BACKGROUND_IMAGE_PATHS = {
							'day': os.path.join(os.getcwd(), 'resources/images/background-day.png'),
							'night': os.path.join(os.getcwd(), 'resources/images/background-night.png')
						}
PIPE_IMAGE_PATHS = {
						'green': os.path.join(os.getcwd(), 'resources/images/pipe-green.png'),
						'red': os.path.join(os.getcwd(), 'resources/images/pipe-red.png')
					}
OTHER_IMAGE_PATHS = {
						'gameover': os.path.join(os.getcwd(), 'resources/images/gameover.png'),
						'message': os.path.join(os.getcwd(), 'resources/images/message.png'),
						'base': os.path.join(os.getcwd(), 'resources/images/base.png')
					}
# the audio paths
AUDIO_PATHS = {
				'die': os.path.join(os.getcwd(), 'resources/audios/die.wav'),
				'hit': os.path.join(os.getcwd(), 'resources/audios/hit.wav'),
				'point': os.path.join(os.getcwd(), 'resources/audios/point.wav'),
				'swoosh': os.path.join(os.getcwd(), 'resources/audios/swoosh.wav'),
				'wing': os.path.join(os.getcwd(), 'resources/audios/wing.wav')
			}

'''define the network'''
class deepQNetwork(nn.Module):
    def __init__(self, imagesize, in_channels=4, num_actions=2, **kwargs):
        super(deepQNetwork, self).__init__()
        assert imagesize == (80, 80), 'imagesize should be (80, 80), or redesign the deepQNetwork please'
        self.convs = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=3),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.fcs = nn.Sequential(nn.Linear(in_features=6400, out_features=512),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(in_features=512, out_features=num_actions))
    '''forward'''
    def forward(self, x):
        x = self.convs(x)
        x = self.fcs(x.reshape(x.size(0), -1))
        return x

'''pipe class'''
class Pipe(pygame.sprite.Sprite):
	def __init__(self, image, position, type_, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = image
		self.rect = self.image.get_rect()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect.left, self.rect.top = position
		self.type_ = type_
		self.used_for_score = False
	@staticmethod
	def randomPipe(image):
		base_y = 0.79 * SCREENHEIGHT
		up_y = int(base_y * 0.2) + random.randrange(0, int(base_y * 0.6 - PIPE_GAP_SIZE))
		return {'top': (SCREENWIDTH+10, up_y-image.get_height()), 'bottom': (SCREENWIDTH+10, up_y+PIPE_GAP_SIZE)}

'''bird class'''
class Bird(pygame.sprite.Sprite):
	def __init__(self, images, idx, position, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.images = images
		self.image = list(images.values())[idx]
		self.rect = self.image.get_rect()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect.left, self.rect.top = position
		# variables required for vertical movement
		self.is_flapped = False
		self.speed = -9
		# variables required for bird status switch
		self.bird_idx = idx
		self.bird_idx_cycle = itertools.cycle([0, 1, 2, 1])
		self.bird_idx_change_count = 0
	'''update bird'''
	def update(self, boundary_values):
		# update the position vertically
		if not self.is_flapped:
			self.speed = min(self.speed+1, 10)
		self.is_flapped = False
		self.rect.top += self.speed
		# determine if the bird dies because it hits the upper and lower boundaries
		is_dead = False
		if self.rect.bottom > boundary_values[1]:
			is_dead = True
			self.rect.bottom = boundary_values[1]
		if self.rect.top < boundary_values[0]:
			self.rect.top = boundary_values[0]
		# simulate wing vibration
		self.bird_idx_change_count += 1
		if self.bird_idx_change_count % 3 == 0:
			self.bird_idx = next(self.bird_idx_cycle)
			self.image = list(self.images.values())[self.bird_idx]
			self.bird_idx_change_count = 0
		return is_dead
	'''set to fly mode'''
	def setFlapped(self):
		self.is_flapped = True
		self.speed = -9
	'''bind to screen'''
	def draw(self, screen):
		screen.blit(self.image, self.rect)

'''The game start interface'''
def startGame(screen, sounds, bird_images, other_images, backgroud_image,  mode):
	base_pos = [0, SCREENHEIGHT*0.79]
	base_diff_bg = other_images['base'].get_width() - backgroud_image.get_width()
	msg_pos = [(SCREENWIDTH-other_images['message'].get_width())/2, SCREENHEIGHT*0.12]
	bird_idx = 0
	bird_idx_change_count = 0
	bird_idx_cycle = itertools.cycle([0, 1, 2, 1])
	bird_pos = [SCREENWIDTH*0.2, (SCREENHEIGHT-list(bird_images.values())[0].get_height())/2]
	bird_y_shift_count = 0
	bird_y_shift_max = 9
	shift = 1
	clock = pygame.time.Clock()
	if mode == 'train':
		return {'bird_pos': bird_pos, 'base_pos': base_pos, 'bird_idx': bird_idx}
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
				pygame.quit()
				sys.exit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
					return {'bird_pos': bird_pos, 'base_pos': base_pos, 'bird_idx': bird_idx}
		sounds['wing'].play()
		bird_idx_change_count += 1
		if bird_idx_change_count % 5 == 0:
			bird_idx = next(bird_idx_cycle)
			bird_idx_change_count = 0
		base_pos[0] = -((-base_pos[0] + 4) % base_diff_bg)
		bird_y_shift_count += 1
		if bird_y_shift_count == bird_y_shift_max:
			bird_y_shift_max = 16
			shift = -1 * shift
			bird_y_shift_count = 0
		bird_pos[-1] = bird_pos[-1] + shift
		screen.blit(backgroud_image, (0, 0))
		screen.blit(list(bird_images.values())[bird_idx], bird_pos)
		screen.blit(other_images['message'], msg_pos)
		screen.blit(other_images['base'], base_pos)
		pygame.display.update()
		clock.tick(FPS)

'''The game over interface'''
def endGame(screen, sounds, showScore, score, number_images, bird, pipe_sprites, backgroud_image, other_images, base_pos,  mode):
	if mode == 'train':
		return
	sounds['die'].play()
	clock = pygame.time.Clock()
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
				pygame.quit()
				sys.exit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
					return
		boundary_values = [0, base_pos[-1]]
		bird.update(boundary_values)
		screen.blit(backgroud_image, (0, 0))
		pipe_sprites.draw(screen)
		screen.blit(other_images['base'], base_pos)
		showScore(screen, score, number_images)
		bird.draw(screen)
		pygame.display.update()
		clock.tick(FPS)


'''parse arguments'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Use dpn to play flappybird')
	parser.add_argument('--mode', dest='mode', help='Choose <train> or <test> please', default='train', type=str)
	parser.add_argument('--resume', dest='resume', help='If mode is <train> and use --resume, check and load the training history', action='store_true')
	args = parser.parse_args()
	return args


'''initialize the game'''
def initGame():
	pygame.init()
	pygame.mixer.init()
	screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	pygame.display.set_caption('BIT Flappy Bird')
	return screen


'''show the game score'''
def showScore(screen, score, number_images):
	digits = list(str(int(score)))
	width = 0
	for d in digits:
		width += number_images.get(d).get_width()
	offset = (SCREENWIDTH - width) / 2
	for d in digits:
		screen.blit(number_images.get(d), (offset, SCREENHEIGHT*0.1))
		offset += number_images.get(d).get_width()


'''the dqn agent'''
class DQNAgent():
    def __init__(self, mode, backuppath, **kwargs):
        self.mode = mode
        self.backuppath = backuppath
        # define the necessary variables
        self.num_actions = 2
        self.num_input_frames = 4
        self.discount_factor = 0.99
        self.num_observes = 3200
        self.num_explores = 3e6
        self.epsilon = 0.1
        self.init_epsilon = 0.1
        self.final_epsilon = 1e-4
        self.replay_memory_size = 5e4
        self.imagesize = (80, 80)
        self.save_interval = 5000
        self.num_iters = 0
        self.replay_memory_record = deque()
        self.max_score = 0
        self.input_image = None
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.batch_size = 32
        # the instanced model
        self.dqn_model = deepQNetwork(imagesize=self.imagesize, in_channels=self.num_input_frames, num_actions=self.num_actions)
        self.dqn_model = self.dqn_model.cuda() if self.use_cuda else self.dqn_model
        # define the optimizer and loss function
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
    '''get the next action'''
    def nextAction(self, reward):
        # some necessary update
        if self.epsilon > self.final_epsilon and self.num_iters > self.num_observes:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.num_explores
        self.num_iters += 1
        # make decision
        if random.random() <= self.epsilon:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                self.dqn_model.eval()
                x = torch.from_numpy(self.input_image).type(self.FloatTensor)
                preds = self.dqn_model(x).view(-1)
                action = preds.argmax().item()
                self.dqn_model.train()
        # train the model if demand
        loss = torch.tensor([0])
        if self.mode == 'train' and self.num_iters > self.num_observes:
            self.optimizer.zero_grad()
            minibatch = random.sample(self.replay_memory_record, self.batch_size)
            states, actions, rewards, states1, is_gameovers = zip(*minibatch)
            states = torch.from_numpy(np.concatenate(states, axis=0)).type(self.FloatTensor)
            actions = torch.from_numpy(np.concatenate(actions, axis=0)).type(self.FloatTensor).view(self.batch_size, self.num_actions)
            rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).type(self.FloatTensor).view(self.batch_size)
            states1 = torch.from_numpy(np.concatenate(states1, axis=0)).type(self.FloatTensor)
            is_gameovers = torch.from_numpy(np.concatenate(is_gameovers, axis=0)).type(self.FloatTensor).view(self.batch_size)
            q_t = self.dqn_model(states1).detach()
            q_t = torch.max(q_t, dim=1)[0]
            loss = self.mse_loss(rewards + (1 - is_gameovers) * (self.discount_factor * q_t),
                                 (self.dqn_model(states) * actions).sum(1))
            loss.backward()
            self.optimizer.step()
            if self.num_iters % self.save_interval == 0:
                self.saveModel(self.backuppath)
        # print some infos
        if self.mode == 'train':
            print('STATE: train, ITER: %s, EPSILON: %s, ACTION: %s, REWARD: %s, LOSS: %s, MAX_SCORE: %s' % (self.num_iters, self.epsilon, action, reward, loss.item(), self.max_score))
        else:
            print('STATE: test, ACTION: %s, MAX_SCORE: %s' % (action, self.max_score))
        return action
    '''load model'''
    def loadModel(self, modelpath):
        if self.mode == 'train':
            print('[INFO]: load checkpoints from %s and %s' % (modelpath, modelpath.replace('pth', 'pkl')))
            model_dict = torch.load(modelpath)
            data_dict = pickle.load(open(modelpath.replace('pth', 'pkl'), 'rb'))
            self.dqn_model.load_state_dict(model_dict['model'])
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.max_score = data_dict['max_score']
            self.epsilon = data_dict['epsilon']
            self.num_iters = data_dict['num_iters']
            self.replay_memory_record = data_dict['replay_memory_record']
        else:
            print('[INFO]: load checkpoints from %s' % modelpath)
            model_dict = torch.load(modelpath)
            self.dqn_model.load_state_dict(model_dict['model'])
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.max_score = 0
            self.epsilon = self.final_epsilon
            self.num_iters = 0
            self.replay_memory_record = deque()
    '''save model'''
    def saveModel(self, modelpath):
        model_dict = {
                        'model': self.dqn_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
        torch.save(model_dict, modelpath)
        data_dict = {
                        'num_iters': self.num_iters,
                        'epsilon': self.epsilon,
                        'replay_memory_record': self.replay_memory_record,
                        'max_score': self.max_score
                    }
        with open(modelpath.replace('pth', 'pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        print('[INFO]: save checkpoints into %s and %s' % (modelpath, modelpath.replace('pth', 'pkl')))
    '''record the necessary information'''
    def record(self, action, reward, score, is_game_running, image):
        # preprocess game frames
        image = self.preprocess(image, self.imagesize)
        # record the scene and corresponding info
        if self.input_image is None:
            self.input_image = np.stack((image,)*self.num_input_frames, axis=2)
            self.input_image = np.transpose(self.input_image, (2, 0, 1))
            self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
        else:
            image = image.reshape(1, 1, image.shape[0], image.shape[1])
            next_input_image = np.append(image, self.input_image[:, :self.num_input_frames-1, :, :], axis=1)
            action = [0, 1] if action else [1, 0]
            self.replay_memory_record.append((self.input_image, np.array(action), np.array([reward]), next_input_image, np.array([int(not is_game_running)])))
            self.input_image = next_input_image
        if len(self.replay_memory_record) > self.replay_memory_size:
            self.replay_memory_record.popleft()
        # record the max score so far
        if score > self.max_score:
            self.max_score = score
    '''preprocess the image'''
    def preprocess(self, image, imagesize):
        image = skimage.color.rgb2gray(image)
        image = skimage.transform.resize(image, imagesize, mode='constant')
        image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
        image = image / 255.0
        return image


'''the main function to be called'''
def main(mode, agent, modelpath):
	screen = initGame()
	# load the necessary game resources
	# --load the game sounds
	sounds = dict()
	for key, value in AUDIO_PATHS.items():
		sounds[key] = pygame.mixer.Sound(value)
	# --load the score digital images
	number_images = dict()
	for key, value in NUMBER_IMAGE_PATHS.items():
		number_images[key] = pygame.image.load(value).convert_alpha()
	# --the pipes
	pipe_images = dict()
	pipe_images['bottom'] = pygame.image.load(random.choice(list(PIPE_IMAGE_PATHS.values()))).convert_alpha()
	pipe_images['top'] = pygame.transform.rotate(pipe_images['bottom'], 180)
	# --the bird images
	bird_images = dict()
	for key, value in BIRD_IMAGE_PATHS[random.choice(list(BIRD_IMAGE_PATHS.keys()))].items():
		bird_images[key] = pygame.image.load(value).convert_alpha()
	# --the background images
	backgroud_image = pygame.image.load(random.choice(list(BACKGROUND_IMAGE_PATHS.values()))).convert_alpha()
	# --other images
	other_images = dict()
	for key, value in OTHER_IMAGE_PATHS.items():
		other_images[key] = pygame.image.load(value).convert_alpha()
	# the start interface of our game
	game_start_info = startGame(screen, sounds, bird_images, other_images, backgroud_image,  mode)
	# enter the main game loop
	score = 0
	bird_pos, base_pos, bird_idx = list(game_start_info.values())
	base_diff_bg = other_images['base'].get_width() - backgroud_image.get_width()
	clock = pygame.time.Clock()
	# --the instanced class of pipe
	pipe_sprites = pygame.sprite.Group()
	for i in range(2):
		pipe_pos = Pipe.randomPipe( pipe_images.get('top'))
		pipe_sprites.add(Pipe(image=pipe_images.get('top'), position=(SCREENWIDTH+200+i*SCREENWIDTH/2, pipe_pos.get('top')[-1]), type_='top'))
		pipe_sprites.add(Pipe(image=pipe_images.get('bottom'), position=(SCREENWIDTH+200+i*SCREENWIDTH/2, pipe_pos.get('bottom')[-1]), type_='bottom'))
	# --the instanced class of bird
	bird = Bird(images=bird_images, idx=bird_idx, position=bird_pos)
	# --whether add the pipe or not
	is_add_pipe = True
	# --whether the game is running or not
	is_game_running = True
	action = 1
	while is_game_running:
		screen.fill(0)
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
				if mode == 'train': agent.saveModel(modelpath)
				pygame.quit()
				sys.exit()
		# --a general reward
		reward = 0.1
		# --use DQNAgent to play flappybird
		if action:
			bird.setFlapped()
			sounds['wing'].play()
		# --check the collide between bird and pipe
		for pipe in pipe_sprites:
			if pygame.sprite.collide_mask(bird, pipe):
				sounds['hit'].play()
				is_game_running = False
				reward = -1
		# --update the bird
		boundary_values = [0, base_pos[-1]]
		is_dead = bird.update(boundary_values)
		if is_dead:
			sounds['hit'].play()
			is_game_running = False
			reward = -1
		# --move the bases to the left to achieve the effect of bird flying forward
		base_pos[0] = -((-base_pos[0] + 4) % base_diff_bg)
		# --move the pipes to the left to achieve the effect of bird flying forward
		flag = False
		for pipe in pipe_sprites:
			pipe.rect.left -= 4
			if pipe.rect.centerx <= bird.rect.centerx and not pipe.used_for_score:
				pipe.used_for_score = True
				score += 0.5
				reward = 1
				if '.5' in str(score):
					sounds['point'].play()
			if pipe.rect.left < 5 and pipe.rect.left > 0 and is_add_pipe:
				pipe_pos = Pipe.randomPipe( pipe_images.get('top'))
				pipe_sprites.add(Pipe(image=pipe_images.get('top'), position=pipe_pos.get('top'), type_='top'))
				pipe_sprites.add(Pipe(image=pipe_images.get('bottom'), position=pipe_pos.get('bottom'), type_='bottom'))
				is_add_pipe = False
			elif pipe.rect.right < 0:
				pipe_sprites.remove(pipe)
				flag = True
		if flag: is_add_pipe = True
		# --get image
		pipe_sprites.draw(screen)
		bird.draw(screen)
		image = pygame.surfarray.array3d(pygame.display.get_surface())
		image = image[:, :int(0.79*SCREENHEIGHT), :]
		# --blit the necessary game elements on the screen
		screen.blit(backgroud_image, (0, 0))
		pipe_sprites.draw(screen)
		screen.blit(other_images['base'], base_pos)
		showScore(screen, score, number_images)
		bird.draw(screen)
		# --record the action and corresponding reward
		agent.record(action, reward, score, is_game_running, image)
		# --make decision
		action = agent.nextAction(reward)
		# --update screen
		pygame.display.update()
		clock.tick(FPS)
	# the end interface of our game
	endGame(screen, sounds, showScore, score, number_images, bird, pipe_sprites, backgroud_image, other_images, base_pos,  mode)


'''run'''
if __name__ == '__main__':
	# parse arguments in command line
	args = parseArgs()
	mode = args.mode.lower()
	assert mode in ['train', 'test'], '--mode should be <train> or <test>'
	# the instanced class of DQNAgent, and the path to save and load model
	if not os.path.exists('checkpoints'):
		os.mkdir('checkpoints')
	modelpath = 'checkpoints/dqn.pth'
	agent = DQNAgent(mode=mode, backuppath=modelpath)
	if os.path.isfile(modelpath):
		if mode == 'test' or (args.resume and mode == 'train'):
			agent.loadModel(modelpath)
	# begin game
	while True:
		main(mode, agent, modelpath)