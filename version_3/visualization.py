import pygame
from environment import GridMuckEnvV2, Entity, Action

# Color constants
COLOR_EMPTY = (240, 240, 240)      
COLOR_TREE = (34, 139, 34)         
COLOR_AGENT = (30, 144, 255)       
COLOR_ZOMBIE = (178, 34, 34)       
COLOR_GRID_LINE = (200, 200, 200)  
COLOR_BG = (50, 50, 50)            
COLOR_TEXT = (255, 255, 255)       
COLOR_HP_GREEN = (0, 200, 0)       
COLOR_HP_RED = (200, 0, 0)         
COLOR_STICK = (218, 165, 32) # Goldenrod for the stick
COLOR_INFO_BG = (30, 30, 30)       
COLOR_SECTION_HDR = (150, 150, 150)

ENTITY_COLORS = {
    Entity.EMPTY: COLOR_EMPTY,
    Entity.TREE: COLOR_TREE,
    Entity.AGENT: COLOR_AGENT,
    Entity.ZOMBIE: COLOR_ZOMBIE,
}

KEY_TO_ACTION = {
    pygame.K_UP: Action.UP, pygame.K_DOWN: Action.DOWN,
    pygame.K_LEFT: Action.LEFT, pygame.K_RIGHT: Action.RIGHT,
    pygame.K_z: Action.UP, pygame.K_s: Action.DOWN,
    pygame.K_q: Action.LEFT, pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.ATTACK, pygame.K_a: Action.ATTACK,
}

class Visualization:
    def __init__(self, cell_size: int = 80):
        self.cell_size = cell_size
        self.visible = False
        self.running = True
        self.screen = None
        self.clock = None
        self.font_large = None
        self.font_med = None
        self.font_small = None
        self.info_panel_width = 200

    def _ensure_initialized(self, env: GridMuckEnvV2) -> None:
        if self.screen is not None: return
        pygame.init()
        grid_pixel_size = env.size * self.cell_size
        self.screen = pygame.display.set_mode((grid_pixel_size + self.info_panel_width, grid_pixel_size))
        pygame.display.set_caption("GridMuck Manual Play")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_med = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)

    def show(self) -> None:
        self.visible = True
        self.running = True

    def render(self, env: GridMuckEnvV2, status_message: str = None) -> None:
        if not self.visible: return
        self._ensure_initialized(env)
        grid_pixel_size = env.size * self.cell_size

        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

        self.screen.fill(COLOR_BG)

        # --- Draw Grid ---
        for x in range(env.size):
            for y in range(env.size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                entity = Entity(int(env.grid[x, y]))
                pygame.draw.rect(self.screen, ENTITY_COLORS.get(entity, COLOR_EMPTY), rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)
                
                if entity != Entity.EMPTY:
                    label = {Entity.TREE: "T", Entity.AGENT: "A", Entity.ZOMBIE: "Z"}[entity]
                    text_surf = self.font_large.render(label, True, COLOR_TEXT)
                    self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

        # --- Info Panel ---
        info_x = grid_pixel_size + 15
        pygame.draw.rect(self.screen, COLOR_INFO_BG, (grid_pixel_size, 0, self.info_panel_width, grid_pixel_size))
        
        y_ptr = 20
        
        # Section: Status
        hdr = self.font_med.render("STATUS", True, COLOR_SECTION_HDR)
        self.screen.blit(hdr, (info_x, y_ptr))
        y_ptr += 30

        for label, hp in [("Agent", env.agent_hp), ("Zombie", env.zombie_hp)]:
            txt = self.font_small.render(f"{label} HP", True, COLOR_TEXT)
            self.screen.blit(txt, (info_x, y_ptr))
            y_ptr += 20
            for i in range(2):
                color = COLOR_HP_GREEN if i < hp else COLOR_HP_RED
                pygame.draw.rect(self.screen, color, (info_x + i * 30, y_ptr, 25, 12))
            y_ptr += 25

        y_ptr += 20
        # Section: Inventory
        hdr = self.font_med.render("INVENTORY", True, COLOR_SECTION_HDR)
        self.screen.blit(hdr, (info_x, y_ptr))
        y_ptr += 30

        stick_label = "[ STICK ]" if env.has_stick else "[ EMPTY ]"
        stick_color = COLOR_STICK if env.has_stick else (100, 100, 100)
        stick_surf = self.font_small.render(stick_label, True, stick_color)
        self.screen.blit(stick_surf, (info_x, y_ptr))
        
        y_ptr += 50
        # Section: Stats
        hdr = self.font_med.render("STATS", True, COLOR_SECTION_HDR)
        self.screen.blit(hdr, (info_x, y_ptr))
        y_ptr += 30
        step_txt = self.font_small.render(f"Steps: {env.current_step}/{env.max_steps}", True, COLOR_TEXT)
        self.screen.blit(step_txt, (info_x, y_ptr))

        # --- Game Over Overlay ---
        if status_message:
            overlay_surf = pygame.Surface((grid_pixel_size // 1.2, 100), pygame.SRCALPHA)
            overlay_surf.fill((0, 0, 0, 220))
            overlay_rect = overlay_surf.get_rect(center=(grid_pixel_size // 2, grid_pixel_size // 2))
            self.screen.blit(overlay_surf, overlay_rect)
            
            msg_surf = self.font_med.render(status_message, True, (255, 215, 0))
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(grid_pixel_size // 2, grid_pixel_size // 2)))

        pygame.display.flip()
        self.clock.tick(30)

    def get_human_action(self):
        if not self.visible: return None
        while self.running and self.visible:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_x, pygame.K_ESCAPE):
                        self.running = False
                        return None
                    if event.key in KEY_TO_ACTION:
                        return KEY_TO_ACTION[event.key].value
            self.clock.tick(30)
        return None

    def close(self):
        pygame.quit()