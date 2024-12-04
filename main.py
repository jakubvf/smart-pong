import random
import pygame
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf


class ModelManager:
    def __init__(self, save_dir='models'):
        self.save_dir = save_dir
        self.best_score = -float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def save_if_better(self, model, score):
        if score > self.best_score:
            self.best_score = score
            model.save(f'{self.save_dir}/best_model.h5')
            print(f"New best model saved with score: {score}")
            return True
        return False

    def load_best_model(self):
        try:
            return tf.keras.models.load_model(f'{self.save_dir}/best_model.h5')
        except:
            return create_model()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states))


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_shape=(5,), activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_game_state(player, ball, ball_speed_x, ball_speed_y, screen_width=800, screen_height=600):
    # Normalized state representation
    state = [
        player.y / screen_height,  # Paddle Y position
        ball.x / screen_width,  # Ball X position
        ball.y / screen_height,  # Ball Y position
        ball_speed_x / abs(ball_speed_x) if ball_speed_x != 0 else 0,  # Direction rather than speed
        ball_speed_y / abs(ball_speed_y) if ball_speed_y != 0 else 0
    ]
    return np.array(state)


def select_action(model, state, epsilon=None, train_frame_counter=0):
    min_epsilon = 0.01
    max_epsilon = 1.0
    epsilon_decay = 50000

    if epsilon is None:
        epsilon = max(min_epsilon, max_epsilon - (train_frame_counter / epsilon_decay))

    if np.random.random() < epsilon:
        return np.random.randint(3)
    else:
        model_input = np.expand_dims(state, axis=0)
        actions = model.predict(model_input, verbose=0)  # Disabled verbose output
        return np.argmax(actions)


def handle_player(player, ball, ball_speed_x, ball_speed_y, model, is_left, player_speed):
    state = get_game_state(player, ball, ball_speed_x, ball_speed_y)
    action = select_action(model, state)
    reward = 0.0

    prev_distance = abs(player.centery - ball.centery)

    # Apply action
    if action == 0:  # Move up
        player.y -= player_speed
    elif action == 2:  # Move down
        player.y += player_speed

    # Constrain paddle position
    player.y = np.clip(player.y, 0, 500)

    # Calculate reward
    new_distance = abs(player.centery - ball.centery)
    reward += (prev_distance - new_distance) * 0.01  # Proportional positioning reward

    # Hitting ball reward
    if player.colliderect(ball):
        reward += 1.0
        if is_left:
            ball.x = player.right
        else:
            ball.x = player.left - ball.width

        ball_speed_x = -ball_speed_x
        ball_speed_y = random.choice([-abs(ball_speed_x), abs(ball_speed_x)])

    return action, reward, ball_speed_x, ball_speed_y


def play_game(model1, model2, screen_width=800, screen_height=600):
    # Initialize game parameters
    player_speed = 10
    ball_speed = 7
    ball_speed_x = random.choice([-ball_speed, ball_speed])
    ball_speed_y = random.choice([-ball_speed, ball_speed])

    # Initialize game objects
    player1 = pygame.Rect(50, screen_height // 2 - 50, 20, 100)
    player2 = pygame.Rect(screen_width - 70, screen_height // 2 - 50, 20, 100)
    ball = pygame.Rect(screen_width // 2 - 10, screen_height // 2 - 10, 20, 20)

    # Initialize game state
    score1 = score2 = 0
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    episode_reward = 0
    enable_drawing = True

    replay_buffer = ReplayBuffer()
    train_frame_counter = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return score1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                enable_drawing = not enable_drawing

        # Handle player actions
        action1, reward1, ball_speed_x, ball_speed_y = handle_player(
            player1, ball, ball_speed_x, ball_speed_y, model1, True, player_speed)
        action2, reward2, ball_speed_x, ball_speed_y = handle_player(
            player2, ball, ball_speed_x, ball_speed_y, model2, False, player_speed)

        # Update ball position
        ball.x += ball_speed_x
        ball.y += ball_speed_y

        # Ball collision with top/bottom
        if ball.y <= 0 or ball.y >= screen_height - 20:
            ball_speed_y = -ball_speed_y
            ball.y = np.clip(ball.y, 0, screen_height - 20)

        # Scoring
        if ball.x <= 0 or ball.x >= screen_width - 20:
            if ball.x <= 0:
                score2 += 1
            else:
                score1 += 1

            # Reset positions
            ball.x = screen_width // 2 - 10
            ball.y = screen_height // 2 - 10
            player1.y = player2.y = screen_height // 2 - 50

            ball_speed_x = random.choice([-ball_speed, ball_speed])
            ball_speed_y = random.choice([-ball_speed, ball_speed])

            if score1 + score2 >= 10:
                return score1

        # Store experiences
        state1 = get_game_state(player1, ball, ball_speed_x, ball_speed_y)
        state2 = get_game_state(player2, ball, -ball_speed_x, ball_speed_y)

        replay_buffer.add(state1, action1, reward1, get_game_state(player1, ball, ball_speed_x, ball_speed_y))
        replay_buffer.add(state2, action2, reward2, get_game_state(player2, ball, -ball_speed_x, ball_speed_y))

        # Training step
        if train_frame_counter % 4 == 0:  # Reduced training frequency
            batch = replay_buffer.sample(batch_size=32)
            if batch:
                states, actions, rewards, next_states = batch
                actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=3)
                model1.fit(states, actions_one_hot, batch_size=32, epochs=1, verbose=0)

        train_frame_counter += 1

        # Rendering
        if enable_drawing:
            screen.fill((0, 0, 0))
            text = font.render(f"{score1} - {score2}", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(screen_width // 2, 50)))
            pygame.draw.rect(screen, (255, 255, 255), player1)
            pygame.draw.rect(screen, (255, 255, 255), player2)
            pygame.draw.ellipse(screen, (255, 255, 255), ball)
            pygame.display.flip()
            clock.tick(60)

    return score1

global screen

def main():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.SCALED)
    pygame.display.set_caption("AI Pong Training")

    model_manager = ModelManager()
    current_model = create_model()
    generation = 0

    try:
        while True:
            print(f"Generation {generation}")
            avg_score = sum(play_game(current_model, current_model) for _ in range(5)) / 5

            if model_manager.save_if_better(current_model, avg_score):
                new_model = model_manager.load_best_model()
                weights = new_model.get_weights()
                for w in weights:
                    w += np.random.normal(0, 0.05, w.shape)  # Reduced mutation rate
                new_model.set_weights(weights)
                current_model = new_model

            generation += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()