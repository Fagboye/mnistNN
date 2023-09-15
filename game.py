import pygame
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('./minst.h5')

# Pygame variables
width, height = 280, 280
white = (255, 255, 255)
black = (0, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32) 

# For drawing
drawing = False
last_pos = (0, 0)

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0]+float(i)/distance*dx)
        y = int(start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

try:
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN:
            color = white
            pygame.draw.circle(screen, color, e.pos, 10)
            last_pos = e.pos
            drawing = True
        elif e.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif e.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.circle(screen, color, e.pos, 10)
                roundline(screen, color, e.pos, last_pos,  10)
                last_pos = e.pos
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                pygame.image.save(screen, 'digit.png')
                img = Image.open('digit.png').convert('L')
                img = img.resize((28, 28))
                img = np.array(img)
                img = img.reshape(1,28, 28,1)
                img = img/255.0
                prediction = model.predict([img])[0]
                result = np.argmax(prediction)
                print(result)
                screen.fill(black)
                text_surface = font.render(str(result), True, (255, 255, 255))  # Render the prediction
                screen.blit(text_surface, (50, 50))
        pygame.display.flip()
        clock.tick(60)
except StopIteration:
    pass

pygame.quit()

