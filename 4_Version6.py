import numpy as np
import pygame
import sys
import math
import random
import json
from typing import List, Tuple, Optional, Dict

# --- Colors ---
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# --- Board and Display Defaults (changeable at runtime) ---
DEFAULT_ROW_COUNT = 6
DEFAULT_COLUMN_COUNT = 7
PLAYER1 = 1
PLAYER2 = 2
EMPTY = 0
SQUARESIZE = 100  # px

# --- Sound Paths ---
MOVE_SOUND_PATH = "move.wav"
WIN_SOUND_PATH = "win.wav"
ERROR_SOUND_PATH = "error.wav"

def load_sound(path: str):
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None

# --- Game Board ---
class GameBoard:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self._hash_cache = None
        self._valid_locations_cache = None

    def reset(self):
        self.board.fill(0)
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._hash_cache = None
        self._valid_locations_cache = None

    def copy(self):
        new_board = GameBoard(self.rows, self.cols)
        new_board.board = self.board.copy()
        return new_board

    def drop_piece(self, row: int, col: int, piece: int) -> bool:
        if self.board[row][col] != EMPTY:
            return False
        self.board[row][col] = piece
        self._invalidate_cache()
        return True

    def is_valid_location(self, col: int) -> bool:
        return 0 <= col < self.cols and self.board[0][col] == EMPTY

    def get_next_open_row(self, col: int) -> Optional[int]:
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                return r
        return None

    def get_valid_locations(self) -> List[int]:
        if self._valid_locations_cache is None:
            self._valid_locations_cache = [
                col for col in range(self.cols)
                if self.is_valid_location(col)
            ]
        return self._valid_locations_cache

    def is_full(self) -> bool:
        return len(self.get_valid_locations()) == 0

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash((self.board.tobytes(), self.rows, self.cols))
        return self._hash_cache

# --- Win Checking ---
class WinChecker:
    @staticmethod
    def check_winning_move(board: np.ndarray, piece: int) -> Optional[List[Tuple[int, int]]]:
        rows, cols = board.shape
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r][c + i] == piece for i in range(4)):
                    return [(r, c + i) for i in range(4)]
        # Vertical
        for c in range(cols):
            for r in range(rows - 3):
                if all(board[r + i][c] == piece for i in range(4)):
                    return [(r + i, c) for i in range(4)]
        # Positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r + i][c + i] == piece for i in range(4)):
                    return [(r + i, c + i) for i in range(4)]
        # Negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r - i][c + i] == piece for i in range(4)):
                    return [(r - i, c + i) for i in range(4)]
        return None

    @staticmethod
    def is_winning_move(game_board: GameBoard, piece: int) -> Optional[List[Tuple[int, int]]]:
        return WinChecker.check_winning_move(game_board.board, piece)

# --- ML Model Stub ---
class MLModelStub:
    def __init__(self):
        # Replace this with your model loading code if available
        pass

    def predict(self, board: np.ndarray, player: int) -> float:
        # Replace with your ML board evaluation
        # For example: flatten board, add player, and call model.predict(...)
        # Should return a score (higher for better for 'player')
        return 0.0

# --- AI Player (Minimax with optional ML) ---
class AIPlayer:
    def __init__(self, difficulty: str = 'Medium', rows: int = DEFAULT_ROW_COUNT, cols: int = DEFAULT_COLUMN_COUNT, ml_model=None):
        self.difficulty = difficulty
        self.depths = {'Easy': 3, 'Medium': 5, 'Hard': 7}
        self.transposition_table = {}
        self.rows = rows
        self.cols = cols
        self.ml_model = ml_model

    def evaluate_window(self, window: List[int], piece: int) -> int:
        opponent_piece = PLAYER1 if piece == PLAYER2 else PLAYER2
        if window.count(piece) == 4:
            return 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            return 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            return 2
        elif window.count(opponent_piece) == 3 and window.count(EMPTY) == 1:
            return -4
        return 0

    def score_position(self, game_board, piece):
        # Use ML model if available
        if self.ml_model is not None:
            return self.ml_model.predict(game_board.board, piece)
        # Classic heuristic
        score = 0
        board = game_board.board
        rows, cols = game_board.rows, game_board.cols
        center_col = cols // 2
        center_array = board[:, center_col]
        score += np.count_nonzero(center_array == piece) * 3
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c:c + 4].tolist()
                score += self.evaluate_window(window, piece)
        for c in range(cols):
            for r in range(rows - 3):
                window = board[r:r + 4, c].tolist()
                score += self.evaluate_window(window, piece)
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, piece)
        return score

    def minimax(self, game_board, depth, alpha, beta, maximizing_player):
        board_key = hash(game_board)
        if board_key in self.transposition_table:
            cached_depth, cached_score, cached_col = self.transposition_table[board_key]
            if cached_depth >= depth:
                return cached_col, cached_score

        valid_locations = game_board.get_valid_locations()
        is_terminal = (WinChecker.is_winning_move(game_board, PLAYER1) is not None or
                       WinChecker.is_winning_move(game_board, PLAYER2) is not None or
                       not valid_locations)

        if depth == 0 or is_terminal:
            if is_terminal:
                if WinChecker.is_winning_move(game_board, PLAYER2):
                    return None, 100000000
                elif WinChecker.is_winning_move(game_board, PLAYER1):
                    return None, -100000000
                else:
                    return None, 0
            else:
                return None, self.score_position(game_board, PLAYER2)

        center = self.cols // 2
        valid_locations.sort(key=lambda x: abs(x - center))
        if center in valid_locations:
            valid_locations = [center] + [x for x in valid_locations if x != center]

        if maximizing_player:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = game_board.get_next_open_row(col)
                if row is not None:
                    board_copy = game_board.copy()
                    board_copy.drop_piece(row, col, PLAYER2)
                    new_score = self.minimax(board_copy, depth - 1, alpha, beta, False)[1]
                    if new_score > value:
                        value = new_score
                        column = col
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            self.transposition_table[board_key] = (depth, value, column)
            return column, value
        else:
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = game_board.get_next_open_row(col)
                if row is not None:
                    board_copy = game_board.copy()
                    board_copy.drop_piece(row, col, PLAYER1)
                    new_score = self.minimax(board_copy, depth - 1, alpha, beta, True)[1]
                    if new_score < value:
                        value = new_score
                        column = col
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
            self.transposition_table[board_key] = (depth, value, column)
            return column, value

    def get_best_move(self, game_board):
        depth = self.depths[self.difficulty]
        valid_locations = game_board.get_valid_locations()
        if not valid_locations:
            return -1
        col, _ = self.minimax(game_board, depth, -math.inf, math.inf, True)
        return col if col is not None else random.choice(valid_locations)

# --- Game Data Logger ---
class GameLogger:
    def __init__(self, path="game_data.jsonl"):
        self.path = path
        self.log = []

    def record_move(self, board: np.ndarray, player: int, move: int):
        self.log.append({
            "board": board.tolist(),
            "player": player,
            "move": move
        })

    def finalize_and_save(self, outcome: str):
        for entry in self.log:
            entry["result"] = outcome
        with open(self.path, "a") as f:
            for entry in self.log:
                f.write(json.dumps(entry) + "\n")
        self.log = []

# --- Game Renderer ---
class GameRenderer:
    def __init__(self, screen, rows: int, cols: int, square_size: int):
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.square = square_size
        self.width = cols * square_size
        self.height = (rows + 1) * square_size
        self.radius = int(square_size / 2 - 5)
        self.font_large = pygame.font.SysFont("Arial", 75)
        self.font_medium = pygame.font.SysFont("Arial", 50)
        self.font_small = pygame.font.SysFont("Arial", 30)
        self._create_board_template()

    def _create_board_template(self):
        self.board_surface = pygame.Surface((self.width, self.height - self.square))
        self.board_surface.fill(BLUE)
        for c in range(self.cols):
            for r in range(self.rows):
                pygame.draw.circle(
                    self.board_surface, BLACK,
                    (int(c * self.square + self.square / 2), int(r * self.square + self.square / 2)),
                    self.radius
                )

    def draw_board(self, game_board: GameBoard):
        self.screen.blit(self.board_surface, (0, self.square))
        for c in range(self.cols):
            for r in range(self.rows):
                piece = game_board.board[r][c]
                if piece != EMPTY:
                    color = RED if piece == PLAYER1 else YELLOW
                    pygame.draw.circle(
                        self.screen, color,
                        (int(c * self.square + self.square / 2), int((r + 1) * self.square + self.square / 2)),
                        self.radius
                    )
        pygame.display.flip()

    def draw_preview_piece(self, pos_x: int, player: int):
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, self.square))
        color = RED if player == PLAYER1 else YELLOW
        pygame.draw.circle(self.screen, color, (pos_x, self.square // 2), self.radius)
        pygame.display.update((0, 0, self.width, self.square))

    def animate_piece_drop(self, game_board: GameBoard, col: int, final_row: int, piece: int):
        color = RED if piece == PLAYER1 else YELLOW
        start_y = self.square // 2
        end_y = int((final_row + 1) * self.square + self.square / 2)
        distance = end_y - start_y
        steps = max(1, distance // 10)
        for step in range(steps + 1):
            progress = step / steps
            eased_progress = progress * progress * progress * (progress * (progress * 6 - 15) + 10)
            current_y = start_y + (distance * eased_progress)
            self.screen.fill(BLACK)
            self.screen.blit(self.board_surface, (0, self.square))
            for r in range(self.rows):
                for c in range(self.cols):
                    existing_piece = game_board.board[r][c]
                    if existing_piece != EMPTY:
                        existing_color = RED if existing_piece == PLAYER1 else YELLOW
                        pygame.draw.circle(
                            self.screen, existing_color,
                            (int(c * self.square + self.square / 2), int((r + 1) * self.square + self.square / 2)),
                            self.radius
                        )
            pygame.draw.circle(self.screen, color, (int(col * self.square + self.square / 2), int(current_y)), self.radius)
            pygame.display.flip()
            pygame.time.wait(10)

    def highlight_win(self, win_cells: List[Tuple[int, int]]):
        for r, c in win_cells:
            pygame.draw.circle(
                self.screen, GREEN,
                (int(c * self.square + self.square / 2), int((r + 1) * self.square + self.square / 2)),
                self.radius, 8
            )
        pygame.display.flip()

    def draw_button(self, x: int, y: int, width: int, height: int, text: str, font) -> pygame.Rect:
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, WHITE, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
        return rect

    def draw_main_menu(self) -> Tuple[pygame.Rect, pygame.Rect]:
        self.screen.fill(BLACK)
        title = self.font_large.render("Connect 4", True, WHITE)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title, title_rect)
        btn_w, btn_h = 250, 70
        x = (self.width - btn_w) // 2
        y1 = self.height // 2 - 100
        y2 = self.height // 2
        pvp_rect = self.draw_button(x, y1, btn_w, btn_h, "Player vs Player", self.font_small)
        pve_rect = self.draw_button(x, y2, btn_w, btn_h, "Player vs AI", self.font_small)
        pygame.display.flip()
        return pvp_rect, pve_rect

    def draw_difficulty_menu(self) -> Tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
        self.screen.fill(BLACK)
        title = self.font_medium.render("Select Difficulty", True, WHITE)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)
        btn_w, btn_h = 150, 50
        easy_x = (self.width - btn_w * 3 - 50) // 2
        med_x = easy_x + btn_w + 25
        hard_x = med_x + btn_w + 25
        btn_y = self.height // 2 - 100
        back_x = (self.width - 100) // 2
        back_y = self.height // 2 + 50
        easy_rect = self.draw_button(easy_x, btn_y, btn_w, btn_h, "Easy", self.font_small)
        medium_rect = self.draw_button(med_x, btn_y, btn_w, btn_h, "Medium", self.font_small)
        hard_rect = self.draw_button(hard_x, btn_y, btn_w, btn_h, "Hard", self.font_small)
        back_rect = self.draw_button(back_x, back_y, 100, 50, "Back", self.font_small)
        pygame.display.flip()
        return easy_rect, medium_rect, hard_rect, back_rect

    def draw_game_over(self, winner: str, is_tie: bool = False) -> pygame.Rect:
        text = "It's a Tie!" if is_tie else f"{winner} Wins!"
        color = WHITE if is_tie else (RED if winner == "Red" else YELLOW)
        label = self.font_large.render(text, True, color)
        self.screen.blit(label, (40, 10))
        replay_rect = self.draw_button((self.width - 150) // 2, self.height // 2 + 150, 150, 50, "Play Again", self.font_small)
        pygame.display.flip()
        return replay_rect

    def draw_stats(self, stats: Dict[str, int]):
        text = f"Red:{stats['Red']}  Yellow:{stats['Yellow']}  Ties:{stats['Ties']}"
        label = self.font_small.render(text, True, WHITE)
        self.screen.blit(label, (10, 10))

# --- Main Game ---
class Connect4Game:
    def __init__(self):
        print("Welcome to Connect 4!")
        try:
            rows = int(input(f"Rows (default {DEFAULT_ROW_COUNT}): ") or DEFAULT_ROW_COUNT)
            cols = int(input(f"Columns (default {DEFAULT_COLUMN_COUNT}): ") or DEFAULT_COLUMN_COUNT)
        except Exception:
            rows, cols = DEFAULT_ROW_COUNT, DEFAULT_COLUMN_COUNT
        self.rows, self.cols = rows, cols
        self.width = cols * SQUARESIZE
        self.height = (rows + 1) * SQUARESIZE
        self.size = (self.width, self.height)
        self.radius = int(SQUARESIZE / 2 - 5)
        pygame.init()
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Connect 4 ML")
        icon = pygame.Surface((32, 32))
        icon.fill(RED)
        pygame.display.set_icon(icon)
        self.clock = pygame.time.Clock()
        self.move_sound = load_sound(MOVE_SOUND_PATH)
        self.win_sound = load_sound(WIN_SOUND_PATH)
        self.error_sound = load_sound(ERROR_SOUND_PATH)
        self.board = GameBoard(self.rows, self.cols)
        self.renderer = GameRenderer(self.screen, self.rows, self.cols, SQUARESIZE)
        self.game_mode = None
        self.current_player = PLAYER1
        self.game_over = False
        self.stats = {'Red': 0, 'Yellow': 0, 'Ties': 0}
        self.ml_model = MLModelStub()  # Replace with your ML model if available
        self.logger = GameLogger()
        self.ai_player = None

    def play_sound(self, sound):
        if sound:
            try:
                sound.play()
            except Exception:
                pass
        else:
            try:
                import platform
                if platform.system() == "Windows":
                    import winsound
                    winsound.MessageBeep()
                else:
                    print('\a')  # ASCII bell
            except Exception:
                pass

    def handle_main_menu(self):
        pvp_rect, pve_rect = self.renderer.draw_main_menu()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pvp_rect.collidepoint(event.pos): return 'pvp'
                    if pve_rect.collidepoint(event.pos): return 'pve'
            self.clock.tick(60)

    def handle_difficulty_selection(self):
        easy, med, hard, back = self.renderer.draw_difficulty_menu()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if easy.collidepoint(event.pos): return 'Easy'
                    if med.collidepoint(event.pos): return 'Medium'
                    if hard.collidepoint(event.pos): return 'Hard'
                    if back.collidepoint(event.pos): return 'back'
            self.clock.tick(60)

    def reset_game(self):
        self.board.reset()
        self.current_player = PLAYER1
        self.game_over = False
        self.logger.log = []
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, SQUARESIZE))
        self.renderer.draw_board(self.board)
        self.renderer.draw_stats(self.stats)
        pygame.display.update()

    def handle_player_move(self, col: int):
        if not self.board.is_valid_location(col):
            self.play_sound(self.error_sound)
            return
        row = self.board.get_next_open_row(col)
        self.renderer.animate_piece_drop(self.board, col, row, self.current_player)
        self.board.drop_piece(row, col, self.current_player)
        self.logger.record_move(self.board.board, self.current_player, col)
        self.play_sound(self.move_sound)
        win_cells = WinChecker.is_winning_move(self.board, self.current_player)
        if win_cells:
            self.renderer.draw_board(self.board)
            self.renderer.highlight_win(win_cells)
            winner = "Red" if self.current_player == PLAYER1 else "Yellow"
            self.stats[winner] += 1
            self.play_sound(self.win_sound)
            self.renderer.draw_game_over(winner)
            self.logger.finalize_and_save(f"{winner} wins")
            self.game_over = True
        elif self.board.is_full():
            self.renderer.draw_board(self.board)
            self.stats['Ties'] += 1
            self.renderer.draw_game_over("", is_tie=True)
            self.logger.finalize_and_save("draw")
            self.game_over = True
        else:
            self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
            self.renderer.draw_board(self.board)
            self.renderer.draw_stats(self.stats)
            pygame.display.update()

    def handle_ai_move(self):
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, SQUARESIZE))
        pygame.display.update((0, 0, self.width, SQUARESIZE))
        pygame.time.wait(500)
        col = self.ai_player.get_best_move(self.board)
        if col == -1:
            return
        row = self.board.get_next_open_row(col)
        self.renderer.animate_piece_drop(self.board, col, row, PLAYER2)
        self.board.drop_piece(row, col, PLAYER2)
        self.logger.record_move(self.board.board, PLAYER2, col)
        self.play_sound(self.move_sound)
        win_cells = WinChecker.is_winning_move(self.board, PLAYER2)
        if win_cells:
            self.renderer.draw_board(self.board)
            self.renderer.highlight_win(win_cells)
            self.stats['Yellow'] += 1
            self.play_sound(self.win_sound)
            self.renderer.draw_game_over("AI")
            self.logger.finalize_and_save("Yellow wins")
            self.game_over = True
        elif self.board.is_full():
            self.renderer.draw_board(self.board)
            self.stats['Ties'] += 1
            self.renderer.draw_game_over("", is_tie=True)
            self.logger.finalize_and_save("draw")
            self.game_over = True
        else:
            self.current_player = PLAYER1
            self.renderer.draw_board(self.board)
            self.renderer.draw_stats(self.stats)
            pygame.display.update()

    def run(self):
        while True:
            self.game_mode = self.handle_main_menu()
            if self.game_mode is None: break

            if self.game_mode == 'pve':
                difficulty = self.handle_difficulty_selection()
                if difficulty is None: break
                if difficulty == 'back': continue
                self.ai_player = AIPlayer(difficulty, self.rows, self.cols, ml_model=self.ml_model)
            else:
                self.ai_player = None

            playing_this_mode = True
            while playing_this_mode:
                self.reset_game()
                while not self.game_over:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            playing_this_mode = False
                            self.game_over = True
                            break
                        # Keyboard controls
                        if event.type == pygame.KEYDOWN:
                            if pygame.K_1 <= event.key <= pygame.K_1 + self.cols - 1:
                                col = event.key - pygame.K_1
                                if self.game_mode == 'pvp' or self.current_player == PLAYER1:
                                    self.handle_player_move(col)
                            elif event.key == pygame.K_ESCAPE:
                                playing_this_mode = False
                                self.game_over = True
                                break
                            elif event.key == pygame.K_RETURN:
                                if self.game_over:
                                    playing_this_mode = False
                                    break
                        if event.type == pygame.MOUSEMOTION:
                            if self.game_mode == 'pvp' or self.current_player == PLAYER1:
                                self.renderer.draw_preview_piece(event.pos[0], self.current_player)
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.game_mode == 'pvp' or self.current_player == PLAYER1:
                                col = event.pos[0] // SQUARESIZE
                                self.handle_player_move(col)
                    if not playing_this_mode: break
                    if not self.game_over and self.game_mode == 'pve' and self.current_player == PLAYER2:
                        self.handle_ai_move()
                    self.clock.tick(60)

                # Game is over, wait for Play Again/Enter, mouse, or Quit
                if playing_this_mode:
                    waiting_for_replay = True
                    while waiting_for_replay:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                waiting_for_replay = False
                                playing_this_mode = False
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                replay_rect = pygame.Rect((self.width - 150) // 2, self.height // 2 + 150, 150, 50)
                                if replay_rect.collidepoint(event.pos):
                                    waiting_for_replay = False
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                                waiting_for_replay = False
                        self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    try:
        game = Connect4Game()
        game.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()