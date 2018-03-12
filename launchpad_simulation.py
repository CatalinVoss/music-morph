import pygame
import sys
import rewards

pygame.init()

MARGIN = 5
WIDTH = 20
HEIGHT = 20

# Tutorial: https://www.cs.ucsb.edu/~pconrad/cs5nm/topics/pygame/drawing/
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
darkBlue = (0,0,128)
white = (255,255,255)
black = (0,0,0)
pink = (255,200,200)

screen = pygame.display.set_mode((480,960))
pygame.display.set_caption('Neural DJ')

def draw_launchpad(color=white):
    for row in range(rewards.NUM_NOTES):
        for column in range(rewards.NUM_OCCURENCES):
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

while (True):
   # Check for quit events
   for event in pygame.event.get():
        if event.type == pygame.QUIT:
             pygame.quit(); sys.exit();

   # Erase the screen
   screen.fill(green)

   # Draw launchpad
   draw_launchpad()

   # # Limit to 60 frames per second
   # clock.tick(60)
   # update the screen
   pygame.display.update()
