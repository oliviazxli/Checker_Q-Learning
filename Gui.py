import pygame
import sys
from Board_operations import check_jump_required, generate_options, update_board


class Gui():
    # Color definitions
    C1 = (255, 255, 255);
    C2 = (0, 0, 0)  # White / Black pieces
    C3 = (230, 225, 225);
    C4 = (40, 40, 40)  # Inner layer for King pieces
    CB = (179, 149, 96);
    CD = (125, 107, 79)  # Board Light / Dark squares
    CS = (167, 229, 211);
    CM = (174, 181, 161)  # Selected / Movable highlight
    CR = (174, 141, 121);
    CL = (174, 181, 181)  # History trajectory / Restricted options
    CZ = (154, 161, 161)

    def __init__(self, board: list, size: tuple, clock: object, screen: object, type_: int) -> None:
        self.type = type_  # Initial assignment, will be overwritten by menu selection
        self.king_type = 3 if self.type == 1 else 4
        self.board = board
        self.selected_block = None
        self.highlighted_blocks = []
        self.limited_options = []
        self.red_blocks = []  # Player movement trajectory
        self.blue_blocks = []  # AI movement trajectory
        self.size = size
        self.clock = clock
        self.screen = screen

        self.board_size = self.size[1]
        self.cell_size = self.board_size / 8

        self.win_messsage = ""
        self.current_turn_text = "Waiting..."

    def draw_start_menu(self) -> int:
        """Draws the start menu for color selection, returns the chosen role code (1 for White, 2 for Black)"""
        selected = None
        while selected is None:
            self.screen.fill((245, 245, 245))
            font_title = pygame.font.SysFont('Arial', 36, bold=True)
            font_btn = pygame.font.SysFont('Arial', 28, bold=True)

            # Title
            title_surf = font_title.render('SELECT YOUR SIDE', True, (40, 40, 40))
            self.screen.blit(title_surf, (self.size[0] // 2 - title_surf.get_width() // 2, 180))

            # Button positions
            white_rect = pygame.Rect(self.size[0] // 2 - 220, 320, 200, 100)
            black_rect = pygame.Rect(self.size[0] // 2 + 20, 320, 200, 100)

            # Draw buttons with shadow effects
            pygame.draw.rect(self.screen, (255, 255, 255), white_rect, border_radius=12)
            pygame.draw.rect(self.screen, (30, 30, 30), black_rect, border_radius=12)
            pygame.draw.rect(self.screen, (200, 200, 200), white_rect, 3, border_radius=12)

            # Button text
            w_text = font_btn.render('WHITE', True, (0, 0, 0))
            b_text = font_btn.render('BLACK', True, (255, 255, 255))
            self.screen.blit(w_text, (
                white_rect.centerx - w_text.get_width() // 2, white_rect.centery - w_text.get_height() // 2))
            self.screen.blit(b_text, (
                black_rect.centerx - b_text.get_width() // 2, black_rect.centery - b_text.get_height() // 2))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if white_rect.collidepoint(event.pos):
                        selected = 1
                    elif black_rect.collidepoint(event.pos):
                        selected = 2

            pygame.display.update()
            self.clock.tick(60)

        self.type = selected
        self.king_type = 3 if self.type == 1 else 4
        return selected

    def choose_action(self) -> tuple:
        move = None
        self.current_turn_text = "Your Turn"
        # Check if any piece is forced to jump
        self.limited_options = check_jump_required(self.board, self.type)

        # If only one piece can move (due to forced jump rule), auto-select it
        if len(self.limited_options) == 1:
            self.selected_block = self.limited_options[0]
            self.highlighted_blocks = generate_options(self.selected_block, self.board, only_jump=True)

        running = True
        while running:
            mouse_click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: mouse_click = True

            if mouse_click:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x, grid_y = int(mouse_x // self.cell_size), int(mouse_y // self.cell_size)

                if grid_x < 8 and grid_y < 8:
                    move_made = False
                    # Check if the user clicked on a highlighted destination square
                    for block in self.highlighted_blocks:
                        if (grid_x, grid_y) == block:
                            move = (self.selected_block, (grid_x, grid_y))
                            hopped = update_board(self.selected_block, (grid_x, grid_y), self.board)
                            self.selected_block, self.highlighted_blocks, self.limited_options = None, [], []

                            # Multi-jump logic: Check if the piece can jump again
                            if hopped and generate_options((grid_x, grid_y), self.board, only_jump=True):
                                return True, move  # Continue the sequence
                            running = False
                            move_made = True
                            break

                    # If no move was made, check if the user is selecting a piece
                    if not move_made:
                        spot = self.board[grid_y][grid_x]
                        if spot == self.type or spot == self.king_type:
                            # If no forced jumps exist, allow selecting any of player's pieces
                            if not self.limited_options:
                                self.selected_block = (grid_x, grid_y)
                                self.highlighted_blocks = generate_options((grid_x, grid_y), self.board)
                            # If forced jumps exist, only allow selecting pieces that can jump
                            elif (grid_x, grid_y) in self.limited_options:
                                self.selected_block = (grid_x, grid_y)
                                self.highlighted_blocks = generate_options((grid_x, grid_y), self.board, only_jump=True)

            self.draw()
            self.clock.tick(60)

        self.current_turn_text = "AI Thinking..."
        return False, move

    def draw(self) -> None:
        self.screen.fill((255, 255, 255))
        block_size = self.cell_size

        # Render the board and pieces
        for y in range(8):
            for x in range(8):
                rect = (x * block_size, y * block_size, block_size, block_size)
                # Determine square color
                c = Gui.CB if (x + y) % 2 == 0 else Gui.CD
                if (x, y) == self.selected_block:
                    c = Gui.CS
                elif (x, y) in self.highlighted_blocks:
                    c = Gui.CM
                elif (x, y) in self.limited_options:
                    c = Gui.CL
                elif (x, y) in self.red_blocks:
                    c = Gui.CR
                elif (x, y) in self.blue_blocks:
                    c = Gui.CZ
                pygame.draw.rect(self.screen, c, rect)

                # Draw pieces
                piece = self.board[y][x]
                if piece != 0:
                    pos = (int(x * block_size + block_size / 2), int(y * block_size + block_size / 2))
                    p_color = Gui.C1 if piece in [1, 3] else Gui.C2
                    pygame.draw.circle(self.screen, p_color, pos, int(block_size / 2 - 10))
                    # Draw inner circles for Kings
                    if piece in [3, 4]:
                        k_color = Gui.C3 if piece == 3 else Gui.C4
                        pygame.draw.circle(self.screen, k_color, pos, int(block_size / 2 - 20))

        # Right-side information panel
        sidebar_x = self.board_size
        pygame.draw.rect(self.screen, (240, 240, 240), (sidebar_x, 0, self.size[0] - sidebar_x, self.size[1]))
        font_text = pygame.font.SysFont('Arial', 20)

        # Labels for Player and AI identity
        player_label = "White (You)" if self.type == 1 else "Black (You)"
        ai_label = "Black (AI)" if self.type == 1 else "White (AI)"

        self.screen.blit(font_text.render(player_label, True, (0, 0, 0)), (sidebar_x + 20, 100))
        self.screen.blit(font_text.render(ai_label, True, (0, 0, 0)), (sidebar_x + 20, 140))
        self.screen.blit(font_text.render(f"Status: {self.current_turn_text}", True, (150, 0, 0)),
                         (sidebar_x + 20, 200))

        # Display Win/Loss/Draw message
        if self.win_messsage:
            win_font = pygame.font.SysFont('Arial', 28, bold=True)
            self.screen.blit(win_font.render(self.win_messsage, True, (200, 0, 0)), (sidebar_x + 20, 300))

        pygame.display.update()