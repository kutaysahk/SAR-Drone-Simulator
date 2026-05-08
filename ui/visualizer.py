import sys
import numpy as np
import pygame
from state.environment import EnvironmentState
from state.drone import DroneState


def show_start_screen():
    pygame.init()
    screen = pygame.display.set_mode((900, 560))
    pygame.display.set_caption("Project Aegis")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("Segoe UI", 44, bold=True)
    body_font = pygame.font.SysFont("Segoe UI", 18)
    button_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
    start_button = pygame.Rect(290, 330, 320, 92)

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_button.collidepoint(mouse_pos):
                    return

        screen.fill((12, 18, 28))
        panel_rect = pygame.Rect(70, 58, 760, 444)
        pygame.draw.rect(screen, (38, 52, 72), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (76, 201, 240), pygame.Rect(panel_rect.x, panel_rect.y, panel_rect.width, 4))

        screen.blit(title_font.render("Project Aegis", True, (240, 247, 255)), (120, 116))
        _draw_wrapped_text(
            screen,
            "Search-and-rescue drone simulator with logical safety inference, frontier pathfinding, and genetic optimization.",
            body_font,
            (177, 190, 205),
            pygame.Rect(124, 184, 660, 62),
        )

        _draw_menu_button(
            screen,
            start_button,
            "Start Simulation",
            "Generate rubble, fire, and hidden survivors.",
            start_button.collidepoint(mouse_pos),
            button_font,
            body_font,
        )

        _draw_wrapped_text(
            screen,
            "During the run: D speeds up, S slows down, Space reveals the full map, Esc returns here.",
            body_font,
            (148, 163, 184),
            pygame.Rect(120, 456, 660, 44),
        )

        pygame.display.flip()
        clock.tick(30)


def _draw_menu_button(screen, rect, title, subtitle, is_hovered, title_font, body_font):
    fill = (45, 64, 89) if is_hovered else (30, 41, 59)
    outline = (96, 165, 250) if is_hovered else (71, 85, 105)
    pygame.draw.rect(screen, fill, rect, border_radius=8)
    pygame.draw.rect(screen, outline, rect, 2, border_radius=8)
    screen.blit(title_font.render(title, True, (248, 250, 252)), (rect.x + 24, rect.y + 16))
    _draw_wrapped_text(
        screen,
        subtitle,
        body_font,
        (203, 213, 225),
        pygame.Rect(rect.x + 24, rect.y + 54, rect.width - 48, rect.height - 62),
    )


def _draw_wrapped_text(screen, text, font, color, rect):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= rect.width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = rect.y
    line_height = font.get_height() + 2
    for line in lines:
        if y + line_height > rect.bottom:
            break
        screen.blit(font.render(line, True, color), (rect.x, y))
        y += line_height


class Visualizer:
    def __init__(self, env_width: int, env_height: int, cell_size: int = 40):
        pygame.init()
        self.cell_size = cell_size
        self.width = env_width
        self.height = env_height
        self.sidebar_width = 260
        self.grid_pixel_width = self.width * self.cell_size
        self.grid_pixel_height = self.height * self.cell_size
        self.visual_map = np.full((env_width, env_height), -1, dtype=int)

        self.screen = pygame.display.set_mode((self.grid_pixel_width + self.sidebar_width, self.grid_pixel_height))
        pygame.display.set_caption("Project Aegis: SAR Drone Simulator")
        self.font = pygame.font.SysFont("Segoe UI", 16)
        self.title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)

        self.COLORS = {
            -1: (24, 31, 42),
             0: (206, 213, 224),
             1: (124, 77, 45),
             2: (239, 68, 68),
             3: (45, 212, 105),
        }
        self.DRONE_COLOR = (250, 204, 21)
        self.DRONE_OUTLINE = (15, 23, 42)
        self.HELP_BEACON_COLOR = (80, 255, 80)
        self.speed_multiplier = 1.0
        self.return_to_menu = False
        self.clock = pygame.time.Clock()

    def render(self, environment: EnvironmentState, drone: DroneState, drone_position=None, scan_radius=None):
        keys = pygame.key.get_pressed()
        show_ground_truth = keys[pygame.K_SPACE]

        self._handle_runtime_events()
        self.screen.fill((15, 23, 42))
        self._draw_grid(environment, show_ground_truth)

        if show_ground_truth:
            self._draw_ground_truth_survivors(environment)

        if (pygame.time.get_ticks() // 350) % 2 == 0:
            for (beacon_x, beacon_y), confidence in drone.rescue_beacons.items():
                center = (
                    beacon_x * self.cell_size + self.cell_size // 2,
                    beacon_y * self.cell_size + self.cell_size // 2,
                )
                radius = max(4, self.cell_size // 2)
                width = 2 if confidence < 1.0 else 3
                pygame.draw.circle(self.screen, self.HELP_BEACON_COLOR, center, radius, width)
                pygame.draw.circle(self.screen, self.HELP_BEACON_COLOR, center, max(2, radius // 3))

        if scan_radius is not None:
            self._draw_scan_radius(drone, scan_radius)

        if drone_position is None:
            drone_position = (drone.x, drone.y)

        drone_px = drone_position[0] * self.cell_size
        drone_py = drone_position[1] * self.cell_size
        drone_rect = pygame.Rect(drone_px + 3, drone_py + 3, self.cell_size - 6, self.cell_size - 6)
        pygame.draw.ellipse(self.screen, self.DRONE_COLOR, drone_rect)
        pygame.draw.ellipse(self.screen, self.DRONE_OUTLINE, drone_rect, 2)

        self._draw_sidebar(drone, show_ground_truth)
        pygame.display.flip()
        self.clock.tick(60)

    def _handle_runtime_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.return_to_menu = True
                elif event.key == pygame.K_d:
                    self.speed_multiplier = min(4.0, self.speed_multiplier + 0.25)
                elif event.key == pygame.K_s:
                    self.speed_multiplier = max(0.25, self.speed_multiplier - 0.25)

    def _draw_grid(self, environment: EnvironmentState, show_ground_truth: bool):
        for x in range(self.width):
            for y in range(self.height):
                cell_state = environment.get_cell_truth(x, y) if show_ground_truth else self.visual_map[x, y]
                color = self.COLORS.get(cell_state, (255, 255, 255))
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (30, 41, 59), rect, 1)

    def animate_move(self, environment: EnvironmentState, drone: DroneState, target_x: int, target_y: int):
        start_x, start_y = drone.x, drone.y
        frames = max(3, int(18 / self.speed_multiplier))
        for frame in range(1, frames + 1):
            if self.return_to_menu:
                return
            t = frame / frames
            eased = t * t * (3 - 2 * t)
            x = start_x + (target_x - start_x) * eased
            y = start_y + (target_y - start_y) * eased
            self.render(environment, drone, drone_position=(x, y))

    def animate_reveal(self, environment: EnvironmentState, drone: DroneState, changed_cells: list):
        if not changed_cells:
            self.render(environment, drone)
            return

        max_radius = 1.5
        frames = max(3, int(16 / self.speed_multiplier))
        pending = set(changed_cells)
        for frame in range(frames + 1):
            if self.return_to_menu:
                return
            radius = max_radius * (frame / frames)
            for x, y in list(pending):
                distance = ((x - drone.x) ** 2 + (y - drone.y) ** 2) ** 0.5
                if distance <= radius:
                    self.visual_map[x, y] = drone.internal_map[x, y]
                    pending.remove((x, y))
            self.render(environment, drone, scan_radius=radius)

        for x, y in pending:
            self.visual_map[x, y] = drone.internal_map[x, y]
        self.render(environment, drone)

    def _draw_scan_radius(self, drone: DroneState, scan_radius: float):
        radius_px = int(scan_radius * self.cell_size)
        if radius_px <= 0:
            return

        center = (
            drone.x * self.cell_size + self.cell_size // 2,
            drone.y * self.cell_size + self.cell_size // 2,
        )
        overlay = pygame.Surface((self.grid_pixel_width, self.grid_pixel_height), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (125, 211, 252, 38), center, radius_px)
        pygame.draw.circle(overlay, (125, 211, 252, 160), center, radius_px, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_sidebar(self, drone: DroneState, show_ground_truth: bool):
        panel_x = self.grid_pixel_width
        panel = pygame.Rect(panel_x, 0, self.sidebar_width, self.grid_pixel_height)
        pygame.draw.rect(self.screen, (15, 23, 42), panel)
        pygame.draw.line(self.screen, (51, 65, 85), (panel_x, 0), (panel_x, self.grid_pixel_height), 2)

        y = 24
        self._blit_text("Project Aegis", panel_x + 24, y, self.title_font, (248, 250, 252))
        y += 38
        self._blit_text("Simulation", panel_x + 24, y, self.font, (125, 211, 252))
        y += 42

        stats = [
            f"Drone: ({drone.x}, {drone.y})",
            f"Speed: {self.speed_multiplier:.2f}x",
            f"Help beacons: {len(drone.rescue_beacons)}",
            f"View: {'ground truth' if show_ground_truth else 'drone memory'}",
        ]
        for stat in stats:
            self._blit_text(stat, panel_x + 24, y, self.font, (203, 213, 225))
            y += 22

        y += 12
        self._blit_text("Legend", panel_x + 24, y, self.title_font, (248, 250, 252))
        y += 30

        legend = [
            ("Unknown", self.COLORS[-1]),
            ("Safe", self.COLORS[0]),
            ("Rubble blocked", self.COLORS[1]),
            ("Fire blocked", self.COLORS[2]),
            ("Survivor", self.COLORS[3]),
        ]
        for label, color in legend:
            pygame.draw.rect(self.screen, color, pygame.Rect(panel_x + 24, y + 4, 16, 16), border_radius=3)
            self._blit_text(label, panel_x + 50, y, self.font, (203, 213, 225))
            y += 23

        y += 12
        self._blit_text("Controls", panel_x + 24, y, self.font, (248, 250, 252))
        y += 22
        controls = [
            "Space: reveal full map",
            "D: speed +0.25x",
            "S: speed -0.25x",
            "Esc: start screen",
        ]
        for control in controls:
            self._blit_text(control, panel_x + 24, y, self.font, (148, 163, 184))
            y += 19

    def _draw_ground_truth_survivors(self, environment: EnvironmentState):
        for survivor_x, survivor_y in environment.survivor_locations:
            center = (
                survivor_x * self.cell_size + self.cell_size // 2,
                survivor_y * self.cell_size + self.cell_size // 2,
            )
            pygame.draw.circle(self.screen, (34, 197, 94), center, max(4, self.cell_size // 2))
            pygame.draw.circle(self.screen, (240, 253, 244), center, max(2, self.cell_size // 4))

    def _blit_text(self, text, x, y, font, color):
        self.screen.blit(font.render(text, True, color), (x, y))
