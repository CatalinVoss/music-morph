import pygame
import sys
from linear_schedule import LinearExploration, LinearSchedule
from multilayer_dqn import MusicQN
from linear_dqn import Linear
from configs.linear import config # multilayer
import rewards
import numpy as np
from midi_output import NeuralDJ
import read_midis
import time
pygame.init()

MARGIN = 5
WIDTH = 35
HEIGHT = 35

PARAMS_PATH = 'results/linear1521002473' #'results/rahulnet' # multilayer1521009279
NUM_BARS = 100
VELOCITY_MULTIPLIER = 90

# Tutorial: https://www.cs.ucsb.edu/~pconrad/cs5nm/topics/pygame/drawing/
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
darkBlue = (0,0,128)
white = (255,255,255)
gray = (180, 180, 180)
black = (0,0,0)
pink = (255,200,200)

screen = pygame.display.set_mode((330,970))
pygame.display.set_caption('Neural DJ')

def box_rect(row, col):
    return [(MARGIN + WIDTH) * col + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT]

def draw_launchpad(state, num_notes, num_occurrences):
    for row in range(num_notes):
        for col in range(num_occurrences):
            color = white if state[row, col] == 1 else gray
            pygame.draw.rect(screen,
                             color,
                             box_rect(row, col))

def get_click_box(mouse_x, mouse_y, num_notes, num_occurrences):
    for row in range(num_notes):
        for col in range(num_occurrences):
            rect = pygame.Rect(*box_rect(row, col))
            if rect.collidepoint(mouse_x, mouse_y):
                return (row, col)
    return None

if __name__ == '__main__':
    env = rewards.MusicEnv()

    model = Linear(env, config) # MusicQN
    model.initialize()
    model.load_params(PARAMS_PATH)

    dj = NeuralDJ(read_midis.NUM_NOTES, read_midis.BAR_QUANT, read_midis.ROLL_WINDOW)
    dj.start_playback()

    # Start in some funny state
    state = np.zeros((env.num_notes, env.num_occurrences))
    # state[14,2] = 1
    # state[14,5] = 1
    # state[21,3] = 1
    # state[21,5] = 2

    # neat solo:
    # state[18,2] = 1


    while (True):
        if dj.bar_buffer.empty():
            action, q_vals = model.get_best_action(env.to_onehot(state))
            print("Taking action: "+str(action))
            if action >= env.num_notes*env.beat_types:
                print("That's an offset!")
            else:
                print("\nNOT AN OFFSET!!!!\n")
            new_state = env.toggle(action, state)
            bar = env.midify(new_state)*VELOCITY_MULTIPLIER
            dj.add_bar(bar)
            state = new_state
        # else:
            # print("Waiting for user input")
            
        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit();
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                mouseClicked = True
                print("Click!")

                box = get_click_box(mouse_x, mouse_y, env.num_notes, env.num_occurrences)
                if box:
                    state[box[0], box[1]] = 1 if state[box[0], box[1]] < 1 else 0
                    print("Clicked box "+str(box))
                # TODO: activate bad boy!

        dj.finish_playback()

        # Erase the screen
        screen.fill(green)

        # Draw launchpad
        draw_launchpad(state, env.num_notes, env.num_occurrences)

        # update the screen
        pygame.display.update()

