import pygame
import numpy as np
from game import Map, Entity, Action

# Color constants
COLOR_EMPTY = (240, 240, 240)      # Light gray
COLOR_TREE = (34, 139, 34)         # Forest green
COLOR_AGENT = (30, 144, 255)       # Dodger blue
COLOR_ZOMBIE = (178, 34, 34)       # Firebrick red
COLOR_GRID_LINE = (200, 200, 200)  # Light gray lines
COLOR_BG = (50, 50, 50)            # Dark gray background
COLOR_TEXT = (255, 255, 255)       # White text
COLOR_HP_GREEN = (0, 200, 0)       # Green for HP
COLOR_HP_RED = (200, 0, 0)         # Red for missing HP
COLOR_STICK = (139, 90, 43)        # Brown for stick indicator
COLOR_INFO_BG = (30, 30, 30)       # Darker background for info panel

# Entity-to-color mapping
ENTITY_COLORS = {
    Entity.EMPTY: COLOR_EMPTY,
    Entity.TREE: COLOR_TREE,
    Entity.AGENT: COLOR_AGENT,
    Entity.ZOMBIE: COLOR_ZOMBIE,
}

# Pygame key to Action mapping
KEY_TO_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_z: Action.UP,
    pygame.K_s: Action.DOWN,
    pygame.K_q: Action.LEFT,
    pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.ATTACK,
    pygame.K_a: Action.ATTACK,
}


class Visualization:
    """Pygame-based visualization for the GridMuck game.
    
    This class is completely separated from the game logic in Map.
    It only reads the Map state for rendering and provides input handling.
    """

    def __init__(self, cell_size: int = 80):
        self.cell_size = cell_size
        self.visible = False
        self.running = True
        self.screen = None
        self.clock = None
        self.font_large = None
        self.font_small = None
        self.info_panel_width = 200

    def _ensure_initialized(self, map_obj: Map) -> None:
        """Lazily initialize Pygame and the display window."""
        if self.screen is not None:
            return

        pygame.init()
        grid_pixel_size = map_obj.size * self.cell_size
        window_width = grid_pixel_size + self.info_panel_width
        window_height = grid_pixel_size

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("GridMuck RL - Version 3")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)

    def show(self) -> None:
        """Enable the visualization window."""
        self.visible = True
        self.running = True

    def hide(self) -> None:
        """Hide (but don't destroy) the visualization window."""
        self.visible = False
        if self.screen is not None:
            pygame.display.set_mode((1, 1))  # Minimize window

    def toggle(self) -> None:
        """Toggle visualization on/off."""
        if self.visible:
            self.hide()
        else:
            self.show()

    def is_visible(self) -> bool:
        return self.visible

    def render(self, map_obj: Map) -> None:
        """Render the current game state to the Pygame window."""
        if not self.visible:
            return

        self._ensure_initialized(map_obj)

        grid_pixel_size = map_obj.size * self.cell_size

        # Handle Pygame events (window close, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.visible = False
                pygame.quit()
                self.screen = None
                return

        # Fill background
        self.screen.fill(COLOR_BG)

        # --- Draw the grid ---
        for x in range(map_obj.size):
            for y in range(map_obj.size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                entity = Entity(int(map_obj.grid[x, y]))
                color = ENTITY_COLORS.get(entity, COLOR_EMPTY)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)

        # --- Draw entity labels ---
        for x in range(map_obj.size):
            for y in range(map_obj.size):
                entity = Entity(int(map_obj.grid[x, y]))
                if entity == Entity.EMPTY:
                    continue

                center_x = y * self.cell_size + self.cell_size // 2
                center_y = x * self.cell_size + self.cell_size // 2

                label = ""
                if entity == Entity.TREE:
                    label = "T"
                elif entity == Entity.AGENT:
                    label = "A"
                elif entity == Entity.ZOMBIE:
                    label = "Z"

                if label:
                    text_surf = self.font_large.render(label, True, COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=(center_x, center_y))
                    self.screen.blit(text_surf, text_rect)

        # --- Draw the info panel ---
        info_x = grid_pixel_size + 10
        info_y = 20

        # Info panel background
        panel_rect = pygame.Rect(
            grid_pixel_size, 0, self.info_panel_width, grid_pixel_size
        )
        pygame.draw.rect(self.screen, COLOR_INFO_BG, panel_rect)

        # Title
        title_surf = self.font_large.render("Game Info", True, COLOR_TEXT)
        self.screen.blit(title_surf, (info_x, info_y))
        info_y += 40

        # Agent HP
        hp_text = f"Agent HP: {map_obj.agent_hp}/2"
        hp_surf = self.font_small.render(hp_text, True, COLOR_TEXT)
        self.screen.blit(hp_surf, (info_x, info_y))
        info_y += 30

        # Draw HP bars (colored rectangles)
        bar_width = 20
        bar_height = 15
        for i in range(2):
            bar_color = COLOR_HP_GREEN if i < map_obj.agent_hp else COLOR_HP_RED
            bar_rect = pygame.Rect(info_x + i * (bar_width + 5), info_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, bar_color, bar_rect)
        info_y += 25

        # Zombie HP
        zombie_text = f"Zombie HP: {map_obj.zombie_hp}/2"
        zombie_surf = self.font_small.render(zombie_text, True, COLOR_TEXT)
        self.screen.blit(zombie_surf, (info_x, info_y))
        info_y += 30

        # Draw zombie HP bars (colored rectangles)
        for i in range(2):
            bar_color = COLOR_HP_GREEN if i < map_obj.zombie_hp else COLOR_HP_RED
            bar_rect = pygame.Rect(info_x + i * (bar_width + 5), info_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, bar_color, bar_rect)
        info_y += 25

        # Stick status
        stick_text = f"Stick: {'Yes' if map_obj.has_stick else 'No'}"
        stick_color = COLOR_STICK if map_obj.has_stick else (100, 100, 100)
        stick_surf = self.font_small.render(stick_text, True, stick_color)
        self.screen.blit(stick_surf, (info_x, info_y))
        info_y += 40

        # Controls reminder
        controls_title = self.font_small.render("Controls:", True, COLOR_TEXT)
        self.screen.blit(controls_title, (info_x, info_y))
        info_y += 25

        controls = [
            "Arrows/ZQSD: Move",
            "Space/A: Attack",
            "V: Toggle Vis",
            "X/ESC: Exit",
        ]
        for ctrl in controls:
            ctrl_surf = self.font_small.render(ctrl, True, (180, 180, 180))
            self.screen.blit(ctrl_surf, (info_x, info_y))
            info_y += 22

        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

    def get_action(self, map_obj: Map):
        """Get an action from the user via Pygame input.
        
        Returns:
            - An int action value if a movement/attack key was pressed.
            - None if no action was taken (e.g., visualization was toggled).
            
        Sets self.running to False if the user wants to exit.
        """
        if not self.visible:
            return None

        self._ensure_initialized(map_obj)

        while self.visible and self.running:
            # Keep rendering so the window stays responsive
            self.render(map_obj)

            # If render() handled a QUIT event, stop waiting
            if not self.visible or not self.running:
                return None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.visible = False
                    pygame.quit()
                    self.screen = None
                    return None

                if event.type == pygame.KEYDOWN:
                    # Toggle visualization
                    if event.key == pygame.K_v:
                        self.toggle()
                        return None  # No action taken, toggle happened

                    # Exit
                    if event.key in (pygame.K_x, pygame.K_ESCAPE):
                        self.running = False
                        return None

                    # Map key to action
                    if event.key in KEY_TO_ACTION:
                        return KEY_TO_ACTION[event.key].value

    def close(self) -> None:
        """Clean up Pygame resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        self.visible = False
        self.running = False
