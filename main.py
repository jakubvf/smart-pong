# A pong game played by AI
import random
import pygame

# Initialize the game
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600), pygame.SCALED)

# Set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the fonts
font = pygame.font.Font(None, 36)

# Set up score
score1 = 0
score2 = 0

# Set up the text
text = font.render(f"{score1} - {score2}", True, WHITE)
text_rect = text.get_rect(center=(400, 50))


# Set up the players
player1 = pygame.Rect(50, 250, 20, 100)
player2 = pygame.Rect(730, 250, 20, 100)

# Set up the ball
ball = pygame.Rect(390, 290, 20, 20)

# Define speeds
player_speed = 5
ball_speed_x = random.choice([-3, 3])
ball_speed_y = 3

# Set up the clock
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player1.y -= player_speed
    if keys[pygame.K_s]:
        player1.y += player_speed
    # Player 2
    if keys[pygame.K_UP]:
        player2.y -= player_speed
    if keys[pygame.K_DOWN]:
        player2.y += player_speed

    # Clamp player position
    player1.y = max(0, player1.y)
    player1.y = min(500, player1.y)
    player2.y = max(0, player2.y)
    player2.y = min(500, player2.y)

    # Update the ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Check for collisions
    if player1.colliderect(ball):
        ball_speed_x = -ball_speed_x
        # Adjust ball position to the right of player1
        ball.x = player1.right
        # Add angle based on where the ball hits the paddle
        offset = (ball.centery - player1.centery) / (player1.height / 2)
        ball_speed_y += offset * 2

    if player2.colliderect(ball):
        ball_speed_x = -ball_speed_x
        # Adjust ball position to the left of player2
        ball.x = player2.left - ball.width
        # Add angle based on where the ball hits the paddle
        offset = (ball.centery - player2.centery) / (player2.height / 2)
        ball_speed_y += offset * 2

    # Bounce off walls
    if ball.y <= 0 or ball.y >= 580:
        ball_speed_y = -ball_speed_y

    # Check for scoring
    if ball.x <= 0 or ball.x >= 780:
        if ball.x <= 0:
            score2 += 1
        if ball.x >= 780:
            score1 += 1
        text = font.render(f"{score1} - {score2}", True, WHITE)

        ball.x = 390
        ball.y = 290
        ball_speed_x = random.choice([-3, 3])
        ball_speed_y = 3


    # Update the screen
    screen.fill(BLACK)
    screen.blit(text, text_rect)

    pygame.draw.rect(screen, WHITE, player1)
    pygame.draw.rect(screen, WHITE, player2)

    pygame.draw.ellipse(screen, WHITE, ball)

    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()