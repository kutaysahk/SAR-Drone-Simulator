"""
ui/visualizer.py
================
Pygame visualizer for Project Aegis.

Resizable window support
-------------------------
* The window is created with pygame.RESIZABLE so the user can drag any edge.
* On every VIDEORESIZE event (and at startup) _recalculate_layout() is called.
  It derives cell_size, font sizes, and all layout constants from the actual
  window dimensions so nothing ever overflows.
* The start screen also scales to the current window size.
* A minimum window size (MIN_W x MIN_H) is enforced so the grid stays legible.
"""

import sys
import numpy as np
import pygame
from state.environment import EnvironmentState
from state.drone import DroneState

# Minimum window dimensions
MIN_W = 640
MIN_H = 360

# Sidebar as a fixed fraction of total window width
SIDEBAR_FRACTION = 0.22   # ~22% of window width

# ---------------------------------------------------------------------------
# Start screen
# ---------------------------------------------------------------------------

def show_start_screen():
    pygame.init()
    info = pygame.display.Info()
    # Start at 90% of the display, capped at a comfortable size
    w = max(MIN_W, min(1100, int(info.current_w * 0.90)))
    h = max(MIN_H, min(660, int(info.current_h * 0.90)))
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    pygame.display.set_caption("Project Aegis")
    clock = pygame.time.Clock()

    while True:
        sw, sh = screen.get_size()
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.VIDEORESIZE:
                sw = max(MIN_W, event.w)
                sh = max(MIN_H, event.h)
                screen = pygame.display.set_mode((sw, sh), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                btn = _start_button_rect(sw, sh)
                if btn.collidepoint(mouse_pos):
                    return

        # Scale fonts to window height
        title_size  = max(18, sh // 14)
        body_size   = max(11, sh // 38)
        button_size = max(13, sh // 26)
        title_font  = pygame.font.SysFont("Segoe UI", title_size, bold=True)
        body_font   = pygame.font.SysFont("Segoe UI", body_size)
        button_font = pygame.font.SysFont("Segoe UI", button_size, bold=True)

        px = int(sw * 0.07)
        py = int(sh * 0.10)
        pw = sw - 2 * px
        ph = sh - 2 * py
        panel_rect = pygame.Rect(px, py, pw, ph)
        btn = _start_button_rect(sw, sh)

        screen.fill((12, 18, 28))
        pygame.draw.rect(screen, (38, 52, 72), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (76, 201, 240),
                         pygame.Rect(panel_rect.x, panel_rect.y, panel_rect.width, 4))

        screen.blit(title_font.render("Project Aegis", True, (240, 247, 255)),
                    (px + int(pw * 0.07), py + int(ph * 0.13)))

        desc_rect = pygame.Rect(px + int(pw * 0.07), py + int(ph * 0.30),
                                int(pw * 0.86), int(ph * 0.15))
        _draw_wrapped_text(
            screen,
            "Search-and-rescue drone simulator with logical safety inference, "
            "frontier pathfinding, and genetic optimization.",
            body_font, (177, 190, 205), desc_rect,
        )

        is_hovered = btn.collidepoint(mouse_pos)
        _draw_menu_button(screen, btn, "Start Simulation",
                          "Generate rubble, fire, and hidden survivors.",
                          is_hovered, button_font, body_font)

        hint_rect = pygame.Rect(px + int(pw * 0.07), py + int(ph * 0.82),
                                int(pw * 0.86), int(ph * 0.12))
        _draw_wrapped_text(
            screen,
            "During the run: D speeds up, S slows down, "
            "Space reveals the full map, Esc returns here.",
            body_font, (148, 163, 184), hint_rect,
        )

        pygame.display.flip()
        clock.tick(30)


def _start_button_rect(sw, sh):
    bw = max(200, int(sw * 0.32))
    bh = max(60,  int(sh * 0.14))
    bx = (sw - bw) // 2
    by = int(sh * 0.55)
    return pygame.Rect(bx, by, bw, bh)


def _draw_menu_button(screen, rect, title, subtitle, is_hovered,
                       title_font, body_font):
    fill    = (45, 64, 89)  if is_hovered else (30, 41, 59)
    outline = (96, 165, 250) if is_hovered else (71, 85, 105)
    pygame.draw.rect(screen, fill,    rect, border_radius=8)
    pygame.draw.rect(screen, outline, rect, 2, border_radius=8)
    screen.blit(title_font.render(title, True, (248, 250, 252)),
                (rect.x + 24, rect.y + 14))
    sub_rect = pygame.Rect(rect.x + 24, rect.y + rect.height // 2,
                           rect.width - 48, rect.height // 2 - 8)
    _draw_wrapped_text(screen, subtitle, body_font, (203, 213, 225), sub_rect)


def _draw_wrapped_text(screen, text, font, color, rect):
    words = text.split()
    lines, current = [], ""
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
    lh = font.get_height() + 2
    for line in lines:
        if y + lh > rect.bottom:
            break
        screen.blit(font.render(line, True, color), (rect.x, y))
        y += lh


# ---------------------------------------------------------------------------
# Simulation visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    def __init__(self, env_width: int, env_height: int,
                 cell_size: int = 15, monitor=None):
        pygame.init()
        self.env_width  = env_width
        self.env_height = env_height
        self.monitor    = monitor
        self.visual_map = np.full((env_width, env_height), -1, dtype=int)

        self.COLORS = {
            -1: (24, 31, 42),
             0: (206, 213, 224),
             1: (124, 77, 45),
             2: (239, 68, 68),
             3: (45, 212, 105),
        }
        self.DRONE_COLOR      = (250, 204, 21)
        self.DRONE_OUTLINE    = (15, 23, 42)
        self.HELP_BEACON_COLOR = (80, 255, 80)

        self.speed_multiplier = 1.0
        self.return_to_menu   = False
        self.clock            = pygame.time.Clock()

        # Open window — let _recalculate_layout pick the initial cell size
        info = pygame.display.Info()
        init_w = max(MIN_W, int(info.current_w * 0.92))
        init_h = max(MIN_H, int(info.current_h * 0.92))
        self.screen = pygame.display.set_mode((init_w, init_h), pygame.RESIZABLE)
        pygame.display.set_caption("Project Aegis: SAR Drone Simulator")

        self._recalculate_layout()

    # ------------------------------------------------------------------
    # Layout  (called at startup and on every resize)
    # ------------------------------------------------------------------

    def _recalculate_layout(self):
        """Derive all size-dependent constants from the current window size."""
        sw, sh = self.screen.get_size()

        self.sidebar_width = max(180, int(sw * SIDEBAR_FRACTION))
        grid_area_w = sw - self.sidebar_width
        grid_area_h = sh

        # Largest integer cell size that fits the grid in the available area
        cs_w = max(4, grid_area_w // self.env_width)
        cs_h = max(4, grid_area_h // self.env_height)
        self.cell_size = min(cs_w, cs_h)

        self.grid_pixel_width  = self.env_width  * self.cell_size
        self.grid_pixel_height = self.env_height * self.cell_size

        # Fonts scale with window height
        fs_normal = max(10, sh // 50)
        fs_title  = max(12, sh // 38)
        self.font       = pygame.font.SysFont("Segoe UI", fs_normal)
        self.title_font = pygame.font.SysFont("Segoe UI", fs_title, bold=True)

        # Sidebar origin (right of grid, centred vertically if grid < window)
        self.sidebar_x = self.grid_pixel_width

    # ------------------------------------------------------------------
    # Public render API
    # ------------------------------------------------------------------

    def render(self, environment: EnvironmentState, drone: DroneState,
               drone_position=None, scan_radius=None):
        keys = pygame.key.get_pressed()
        show_ground_truth = keys[pygame.K_SPACE]

        self._handle_runtime_events()
        self.screen.fill((15, 23, 42))
        self._draw_grid(environment, show_ground_truth)

        if show_ground_truth:
            self._draw_ground_truth_survivors(environment)

        # Rescue beacons (blinking)
        if (pygame.time.get_ticks() // 350) % 2 == 0:
            for (bx, by), confidence in drone.rescue_beacons.items():
                cx = bx * self.cell_size + self.cell_size // 2
                cy = by * self.cell_size + self.cell_size // 2
                r  = max(3, self.cell_size // 2)
                w  = 2 if confidence < 1.0 else 3
                pygame.draw.circle(self.screen, self.HELP_BEACON_COLOR, (cx, cy), r, w)
                pygame.draw.circle(self.screen, self.HELP_BEACON_COLOR, (cx, cy), max(2, r // 3))

        if scan_radius is not None:
            self._draw_scan_radius(drone, scan_radius)

        # Drone sprite
        if drone_position is None:
            drone_position = (drone.x, drone.y)
        dpx = drone_position[0] * self.cell_size
        dpy = drone_position[1] * self.cell_size
        pad = max(1, self.cell_size // 5)
        drone_rect = pygame.Rect(dpx + pad, dpy + pad,
                                  self.cell_size - 2 * pad,
                                  self.cell_size - 2 * pad)
        pygame.draw.ellipse(self.screen, self.DRONE_COLOR,   drone_rect)
        pygame.draw.ellipse(self.screen, self.DRONE_OUTLINE, drone_rect, 2)

        self._draw_sidebar(drone, show_ground_truth)
        pygame.display.flip()
        self.clock.tick(60)

    # ------------------------------------------------------------------
    # Animation helpers
    # ------------------------------------------------------------------

    def animate_move(self, environment, drone, target_x, target_y):
        start_x, start_y = drone.x, drone.y
        frames = max(3, int(18 / self.speed_multiplier))
        for frame in range(1, frames + 1):
            if self.return_to_menu:
                return
            t = frame / frames
            eased = t * t * (3 - 2 * t)
            self.render(environment, drone,
                        drone_position=(start_x + (target_x - start_x) * eased,
                                        start_y + (target_y - start_y) * eased))

    def animate_reveal(self, environment, drone, changed_cells):
        if not changed_cells:
            self.render(environment, drone)
            return
        max_radius = 1.5
        frames  = max(3, int(16 / self.speed_multiplier))
        pending = set(changed_cells)
        for frame in range(frames + 1):
            if self.return_to_menu:
                return
            radius = max_radius * (frame / frames)
            for x, y in list(pending):
                if ((x - drone.x) ** 2 + (y - drone.y) ** 2) ** 0.5 <= radius:
                    self.visual_map[x, y] = drone.internal_map[x, y]
                    pending.discard((x, y))
            self.render(environment, drone, scan_radius=radius)
        for x, y in pending:
            self.visual_map[x, y] = drone.internal_map[x, y]
        self.render(environment, drone)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_runtime_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.VIDEORESIZE:
                nw = max(MIN_W, event.w)
                nh = max(MIN_H, event.h)
                self.screen = pygame.display.set_mode((nw, nh), pygame.RESIZABLE)
                self._recalculate_layout()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.return_to_menu = True
                elif event.key == pygame.K_d:
                    self.speed_multiplier = min(4.0, self.speed_multiplier + 0.25)
                elif event.key == pygame.K_s:
                    self.speed_multiplier = max(0.25, self.speed_multiplier - 0.25)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_grid(self, environment, show_ground_truth):
        cs = self.cell_size
        for x in range(self.env_width):
            for y in range(self.env_height):
                val   = (environment.get_cell_truth(x, y)
                         if show_ground_truth else self.visual_map[x, y])
                color = self.COLORS.get(val, (255, 255, 255))
                rect  = pygame.Rect(x * cs, y * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)
                if cs > 6:   # skip grid lines when cells are tiny
                    pygame.draw.rect(self.screen, (30, 41, 59), rect, 1)

    def _draw_scan_radius(self, drone, scan_radius):
        radius_px = int(scan_radius * self.cell_size)
        if radius_px <= 0:
            return
        center = (drone.x * self.cell_size + self.cell_size // 2,
                  drone.y * self.cell_size + self.cell_size // 2)
        overlay = pygame.Surface(
            (self.grid_pixel_width, self.grid_pixel_height), pygame.SRCALPHA
        )
        pygame.draw.circle(overlay, (125, 211, 252, 38),  center, radius_px)
        pygame.draw.circle(overlay, (125, 211, 252, 160), center, radius_px, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_ground_truth_survivors(self, environment):
        cs = self.cell_size
        for sx, sy in environment.survivor_locations:
            cx = sx * cs + cs // 2
            cy = sy * cs + cs // 2
            pygame.draw.circle(self.screen, (34, 197, 94),  (cx, cy), max(3, cs // 2))
            pygame.draw.circle(self.screen, (240, 253, 244), (cx, cy), max(2, cs // 4))

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _draw_sidebar(self, drone, show_ground_truth):
        px  = self.sidebar_x
        sw  = self.sidebar_width
        sh  = self.screen.get_height()
        pad = 14

        # Background + divider
        pygame.draw.rect(self.screen, (15, 23, 42), pygame.Rect(px, 0, sw, sh))
        pygame.draw.line(self.screen, (51, 65, 85), (px, 0), (px, sh), 2)

        y = pad
        self._bt("Project Aegis", px + pad, y, self.title_font, (248, 250, 252))
        y += self.title_font.get_height() + 4
        self._bt("Simulation", px + pad, y, self.font, (125, 211, 252))
        y += self.font.get_height() + 10

        lh = self.font.get_height() + 3
        for stat in [
            f"Drone: ({drone.x}, {drone.y})",
            f"Speed: {self.speed_multiplier:.2f}x",
            f"Beacons: {len(drone.rescue_beacons)}",
            f"View: {'truth' if show_ground_truth else 'memory'}",
        ]:
            self._bt(stat, px + pad, y, self.font, (203, 213, 225))
            y += lh

        y += 6
        pygame.draw.line(self.screen, (51, 65, 85),
                         (px + 8, y), (px + sw - 8, y), 1)
        y += 8
        self._bt("Legend", px + pad, y, self.title_font, (248, 250, 252))
        y += self.title_font.get_height() + 4

        swatch = max(8, self.font.get_height() - 2)
        for label, color in [
            ("Unknown",  self.COLORS[-1]),
            ("Safe",     self.COLORS[0]),
            ("Rubble",   self.COLORS[1]),
            ("Fire",     self.COLORS[2]),
            ("Survivor", self.COLORS[3]),
        ]:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(px + pad, y + 2, swatch, swatch),
                             border_radius=2)
            self._bt(label, px + pad + swatch + 6, y, self.font, (203, 213, 225))
            y += lh

        y += 6
        pygame.draw.line(self.screen, (51, 65, 85),
                         (px + 8, y), (px + sw - 8, y), 1)
        y += 8
        self._bt("Controls", px + pad, y, self.title_font, (248, 250, 252))
        y += self.title_font.get_height() + 2
        for ctrl in ["Space: full map", "D: faster", "S: slower", "Esc: menu"]:
            self._bt(ctrl, px + pad, y, self.font, (148, 163, 184))
            y += lh

        # Live monitoring panel (Pillar 4)
        if self.monitor and self.monitor.step_log and y + 60 < sh:
            self._draw_monitoring_panel(px, y + 8, sw, sh, drone)

    def _draw_monitoring_panel(self, px, start_y, sw, sh, drone):
        mon  = self.monitor
        last = mon.step_log[-1]
        pad  = 14
        lh   = self.font.get_height() + 3
        bar_w = sw - pad * 2
        bar_h = max(5, self.font.get_height() // 2)
        y = start_y

        pygame.draw.line(self.screen, (51, 65, 85),
                         (px + 8, y), (px + sw - 8, y), 1)
        y += 8
        self._bt("Monitoring", px + pad, y, self.title_font, (248, 250, 252))
        y += self.title_font.get_height() + 4

        # Exploration bar
        if y + bar_h + lh > sh:
            return
        filled = int(last.exploration_fraction * bar_w)
        pygame.draw.rect(self.screen, (38, 52, 72),
                         pygame.Rect(px + pad, y, bar_w, bar_h), border_radius=3)
        if filled > 0:
            pygame.draw.rect(self.screen, (56, 189, 248),
                             pygame.Rect(px + pad, y, filled, bar_h), border_radius=3)
        y += bar_h + 2
        self._bt(
            f"Explored {last.exploration_fraction*100:.0f}%  "
            f"({last.total_explored}/{last.traversable_total})",
            px + pad, y, self.font, (125, 211, 252)
        )
        y += lh

        # Survivor recall bar
        if y + bar_h + lh > sh:
            return
        total_s = len(mon._env.survivor_locations)
        found_s = len(drone.rescue_beacons)
        recall  = found_s / total_s if total_s else 0.0
        filled2 = int(recall * bar_w)
        pygame.draw.rect(self.screen, (38, 52, 72),
                         pygame.Rect(px + pad, y, bar_w, bar_h), border_radius=3)
        if filled2 > 0:
            pygame.draw.rect(self.screen, (52, 211, 153),
                             pygame.Rect(px + pad, y, filled2, bar_h), border_radius=3)
        y += bar_h + 2
        self._bt(
            f"Recall {recall*100:.0f}%  ({found_s}/{total_s})",
            px + pad, y, self.font, (52, 211, 153)
        )
        y += lh

        # Key metrics — only draw rows that still fit
        for label, value in [
            ("Steps",    str(last.step)),
            ("KB facts", str(last.kb_fact_count)),
            ("E[U]",     f"{last.decision_utility:.3f}"),
            ("Cov/step", f"{last.coverage_rate:.1f}"),
            ("Battery",  str(last.battery_remaining)),
        ]:
            if y + lh > sh:
                break
            self._bt(f"{label}:", px + pad, y, self.font, (148, 163, 184))
            self._bt(value, px + pad + int(sw * 0.45), y, self.font, (203, 213, 225))
            y += lh

        # Sparkline — only if there's room for at least 20px height
        spark_h = 24
        history = [r.coverage_rate for r in mon.step_log[-30:]]
        if len(history) > 2 and y + spark_h + lh + 4 <= sh:
            y += 4
            self._bt("Coverage rate", px + pad, y, self.font, (148, 163, 184))
            y += lh
            spark_rect = pygame.Rect(px + pad, y, bar_w, spark_h)
            pygame.draw.rect(self.screen, (24, 33, 47), spark_rect, border_radius=3)
            max_val = max(history) if max(history) > 0 else 1
            pts = [
                (
                    px + pad + int(i / (len(history) - 1) * (bar_w - 2)),
                    y + spark_h - 2 - int((v / max_val) * (spark_h - 4)),
                )
                for i, v in enumerate(history)
            ]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (56, 189, 248), False, pts, 2)

    def _bt(self, text, x, y, font, color):
        """Blit text, clipping to the right edge of the sidebar."""
        self.screen.blit(font.render(text, True, color), (x, y))
