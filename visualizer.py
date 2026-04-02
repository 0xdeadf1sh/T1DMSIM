"""
T1DM Simulator — Interactive Visualizer (Pygame)
==================================================
Controls:
  SPACE       — Generate next 24 hours
  R           — Reseed with random seed
  0-9         — Reseed with that digit as seed
  UP/DOWN     — Scroll through curves vertically
  LEFT/RIGHT  — Scroll timeline
  HOME        — Jump to start
  END         — Jump to end
  +/-         — Zoom in/out on time axis
  1-6         — Toggle individual curve visibility
  A           — Toggle all curves on/off
  S           — Screenshot (saves PNG)
  Q / ESC     — Quit

Curves (toggle with number keys):
  1 — Blood Glucose (observed)
  2 — Carb intake curve
  3 — Insulin curve
  4 — Insulin Sensitivity
  5 — Exercise curve
  6 — BG Delta
"""

import sys
import os
import time
import numpy as np

# Force X11 (XWayland) to avoid Wayland flickering with pygame/SDL2
if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ['SDL_VIDEODRIVER'] = 'x11'

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pygame.freetype

from simulator import T1DMSimulator, DT_MINUTES

# ============================================================================
# VISUAL THEME
# ============================================================================

# Colors
BG_COLOR        = (15, 15, 20)
PANEL_COLOR     = (22, 22, 30)
GRID_COLOR      = (35, 35, 45)
GRID_COLOR_MAJOR= (50, 50, 65)
TEXT_COLOR       = (190, 195, 210)
TEXT_DIM         = (100, 105, 120)
TEXT_BRIGHT      = (230, 235, 245)
ACCENT           = (90, 140, 255)

# Curve colors
COLOR_BG_TRACE   = (50, 205, 100)    # Green for BG
COLOR_BG_OBS     = (50, 205, 100)    # Green for CGM reading
COLOR_CARB       = (255, 160, 40)    # Orange for carbs
COLOR_INSULIN    = (80, 150, 255)    # Blue for insulin
COLOR_IS         = (200, 120, 255)   # Purple for IS
COLOR_EXERCISE   = (255, 80, 120)    # Pink/red for exercise
COLOR_DELTA      = (120, 220, 220)   # Cyan for delta

# BG zone colors (for background shading)
COLOR_LOW        = (180, 40, 40, 25)
COLOR_IN_RANGE   = (40, 120, 40, 15)
COLOR_HIGH       = (180, 140, 20, 20)
COLOR_VERY_HIGH  = (180, 40, 40, 20)

# Layout
SIDEBAR_WIDTH    = 280
HEADER_HEIGHT    = 50
FOOTER_HEIGHT    = 30
CHART_PADDING    = 10
MIN_WINDOW_W     = 1200
MIN_WINDOW_H     = 700

STEPS_PER_DAY = 24 * 60 // DT_MINUTES  # 288


# ============================================================================
# CURVE DEFINITIONS
# ============================================================================

CURVES = [
    {'key': 'bg_observed', 'name': 'Blood Glucose',     'color': COLOR_BG_OBS,  'unit': 'mg/dL', 'y_min': 30,  'y_max': 400, 'toggle_key': pygame.K_1},
    {'key': 'total_carb',  'name': 'Carb Intake',       'color': COLOR_CARB,    'unit': 'g/step','y_min': 0,   'y_max': 20,   'toggle_key': pygame.K_2},
    {'key': 'total_insulin','name': 'Insulin',           'color': COLOR_INSULIN, 'unit': 'U/step','y_min': 0,   'y_max': 2,   'toggle_key': pygame.K_3},
    {'key': 'insulin_resistance','name': 'Insulin Resistance','color': COLOR_IS,'unit': '×',  'y_min': 0,   'y_max': 3,   'toggle_key': pygame.K_4},
    {'key': 'total_exercise','name': 'Exercise',         'color': COLOR_EXERCISE,'unit': 'g/step','y_min': 0,   'y_max': 10,   'toggle_key': pygame.K_5},
    {'key': 'bg_delta',    'name': 'BG Delta',           'color': COLOR_DELTA,   'unit': 'mg/dL', 'y_min': -20, 'y_max': 10,  'toggle_key': pygame.K_6},
]


# ============================================================================
# HELPER: Draw text
# ============================================================================

def draw_text(surface, font, text, x, y, color=TEXT_COLOR, anchor='topleft'):
    """Draw text with anchor support."""
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(**{anchor: (x, y)})
    surface.blit(rendered, rect)
    return rect


def format_time(step_idx):
    """Convert step index to HH:MM string."""
    total_min = step_idx * DT_MINUTES
    hours = (total_min // 60) % 24
    minutes = total_min % 60
    return f"{hours:02d}:{minutes:02d}"


def format_day_time(step_idx):
    """Convert step index to Day N HH:MM."""
    total_min = step_idx * DT_MINUTES
    day = total_min // (24 * 60)
    hours = (total_min // 60) % 24
    minutes = total_min % 60
    return f"Day {day + 1}  {hours:02d}:{minutes:02d}"


# ============================================================================
# MAIN VISUALIZER
# ============================================================================

class Visualizer:
    def __init__(self):
        pygame.init()

        # Display
        info = pygame.display.Info()
        self.win_w = max(MIN_WINDOW_W, info.current_w - 100)
        self.win_h = max(MIN_WINDOW_H, info.current_h - 100)
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h),
            pygame.RESIZABLE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("T1DM Simulator")

        # Off-screen buffer to eliminate flickering
        self.buffer = pygame.Surface((self.win_w, self.win_h))

        # Fonts
        self.font_sm = pygame.font.SysFont("DejaVu Sans Mono", 12)
        self.font_md = pygame.font.SysFont("DejaVu Sans Mono", 14)
        self.font_lg = pygame.font.SysFont("DejaVu Sans Mono", 18)
        self.font_xl = pygame.font.SysFont("DejaVu Sans Mono", 22)

        # Simulator
        self.seed = 42
        self.sim = T1DMSimulator(seed=self.seed)
        self.data = None
        self.total_steps = 0

        # View state
        self.scroll_x = 0           # Leftmost visible step index
        self.pixels_per_step = 4.0   # Zoom level
        self.curve_visible = [True, True, True, True, False, False]  # Which curves are shown
        self.hovered_step = None     # Step under mouse cursor

        # Generate initial data
        self._generate(24)

        # Clock
        self.clock = pygame.time.Clock()

    def _generate(self, hours):
        """Generate more data."""
        new_data = self.sim.generate_hours(hours)
        if self.data is None:
            self.data = {k: np.array(v) for k, v in new_data.items()}
        else:
            self.data = {k: np.concatenate([self.data[k], new_data[k]]) for k in self.data}
        self.total_steps = len(self.data['bg'])

    def _reseed(self, seed):
        """Reset with new seed."""
        self.seed = seed
        self.sim = T1DMSimulator(seed=seed)
        self.data = None
        self.total_steps = 0
        self.scroll_x = 0
        self._generate(24)

    def _chart_rect(self):
        """Get the chart drawing area."""
        x = SIDEBAR_WIDTH + CHART_PADDING
        y = HEADER_HEIGHT + CHART_PADDING
        w = self.win_w - SIDEBAR_WIDTH - CHART_PADDING * 2
        h = self.win_h - HEADER_HEIGHT - FOOTER_HEIGHT - CHART_PADDING * 2
        return pygame.Rect(x, y, w, h)

    def _visible_range(self):
        """Get the range of step indices currently visible."""
        chart = self._chart_rect()
        visible_steps = int(chart.width / self.pixels_per_step)
        start = max(0, self.scroll_x)
        end = min(self.total_steps, start + visible_steps)
        return start, end, visible_steps

    def _step_to_x(self, step, chart):
        """Convert a step index to pixel x coordinate."""
        return chart.x + (step - self.scroll_x) * self.pixels_per_step

    def _draw_sidebar(self):
        """Draw the parameter panel on the left."""
        sidebar = pygame.Rect(0, 0, SIDEBAR_WIDTH, self.win_h)
        pygame.draw.rect(self.buffer, PANEL_COLOR, sidebar)
        pygame.draw.line(self.buffer, GRID_COLOR_MAJOR,
                         (SIDEBAR_WIDTH - 1, 0), (SIDEBAR_WIDTH - 1, self.win_h))

        x, y = 12, 12
        draw_text(self.buffer, self.font_lg, "T1DM Simulator", x, y, TEXT_BRIGHT)
        y += 30

        # Seed
        draw_text(self.buffer, self.font_md, f"Seed: {self.seed}", x, y, ACCENT)
        y += 25

        # Patient profile
        draw_text(self.buffer, self.font_md, "— Patient Profile —", x, y, TEXT_DIM)
        y += 22

        p = self.sim.patient
        profile_items = [
            ("Diet Discipline", p.dietary_discipline, COLOR_CARB),
            ("Attentiveness",   p.attentiveness,      COLOR_INSULIN),
            ("Dose Competence", p.dosing_competence,   COLOR_IS),
            ("Consistency",     p.lifestyle_consistency, COLOR_EXERCISE),
        ]
        for label, val, color in profile_items:
            draw_text(self.buffer, self.font_sm, label, x, y, TEXT_DIM)
            # Skill bar
            bar_x = x + 140
            bar_w = 100
            bar_h = 10
            bar_y_c = y + 6
            pygame.draw.rect(self.buffer, GRID_COLOR, (bar_x, bar_y_c, bar_w, bar_h))
            fill_w = int(val * bar_w)
            pygame.draw.rect(self.buffer, color, (bar_x, bar_y_c, fill_w, bar_h))
            draw_text(self.buffer, self.font_sm, f"{val:.2f}", bar_x + bar_w + 5, y, TEXT_DIM)
            y += 18

        y += 10
        draw_text(self.buffer, self.font_md, "— Parameters —", x, y, TEXT_DIM)
        y += 22

        summary = self.sim.get_patient_summary()
        param_keys = ['is_base', 'icr', 'correction_factor', 'basal_dose',
                      'cgm_check_interval', 'patience_time', 'exercise_prob',
                      'basal_miss_prob', 'slow_carb_pref', 'panic_factor']
        param_labels = ['IS Base', 'ICR', 'Correction Factor', 'Basal Dose',
                        'CGM Check Interval', 'Patience Time', 'Exercise Prob',
                        'Basal Miss Prob', 'Slow Carb Pref', 'Panic Factor']

        for label, key in zip(param_labels, param_keys):
            draw_text(self.buffer, self.font_sm, f"{label}:", x, y, TEXT_DIM)
            draw_text(self.buffer, self.font_sm, summary[key], x + 150, y, TEXT_COLOR)
            y += 16

        # Stats
        if self.data is not None and self.total_steps > 0:
            y += 15
            draw_text(self.buffer, self.font_md, "— Statistics —", x, y, TEXT_DIM)
            y += 22

            bg = self.data['bg'][:self.total_steps]
            tir = np.mean((bg >= 70) & (bg <= 180)) * 100
            tbr = np.mean(bg < 70) * 100
            tar = np.mean(bg > 180) * 100

            stats = [
                ("Total Time", f"{self.total_steps * DT_MINUTES / 60:.0f}h ({self.total_steps * DT_MINUTES / 1440:.1f}d)"),
                ("Mean BG", f"{bg.mean():.0f} mg/dL"),
                ("BG Range", f"{bg.min():.0f}–{bg.max():.0f}"),
                ("Time in Range", f"{tir:.1f}%"),
                ("Time Below", f"{tbr:.1f}%"),
                ("Time Above", f"{tar:.1f}%"),
            ]

            if self.sim.state.is_sick:
                stats.append(("Status", "SICK"))
            if self.sim.state.is_rare_event_day:
                stats.append(("Today", "RARE EVENT"))

            for label, val in stats:
                draw_text(self.buffer, self.font_sm, f"{label}:", x, y, TEXT_DIM)
                color = TEXT_COLOR
                if label == "Time in Range":
                    color = COLOR_BG_TRACE if tir > 70 else (COLOR_CARB if tir > 40 else (200, 60, 60))
                elif label == "Status":
                    color = (255, 80, 80)
                elif label == "Today":
                    color = (255, 200, 50)
                draw_text(self.buffer, self.font_sm, val, x + 120, y, color)
                y += 16

        # Curve legend / toggles
        y += 15
        draw_text(self.buffer, self.font_md, "— Curves (1-6) —", x, y, TEXT_DIM)
        y += 22

        for i, curve in enumerate(CURVES):
            prefix = "●" if self.curve_visible[i] else "○"
            color = curve['color'] if self.curve_visible[i] else TEXT_DIM
            draw_text(self.buffer, self.font_sm, f"[{i+1}] {prefix} {curve['name']}", x, y, color)
            y += 16

        # Controls
        y += 15
        draw_text(self.buffer, self.font_md, "— Controls —", x, y, TEXT_DIM)
        y += 20
        controls = [
            "SPACE  Generate +24h",
            "R      Random reseed",
            "0-9    Seed by digit",
            "←→     Scroll time",
            "+−     Zoom",
            "HOME   Jump to start",
            "END    Jump to end",
            "A      Toggle all curves",
            "S      Screenshot",
            "Q/ESC  Quit",
        ]
        for line in controls:
            if y + 14 > self.win_h - 10:
                break
            draw_text(self.buffer, self.font_sm, line, x, y, TEXT_DIM)
            y += 14

    def _draw_bg_zones(self, chart):
        """Draw colored background zones for BG ranges."""
        if not self.curve_visible[0]:
            return

        curve_def = CURVES[0]
        y_min, y_max = curve_def['y_min'], curve_def['y_max']
        h = chart.height

        zones = [
            (30, 54, (180, 30, 30, 20)),      # Very low — red
            (54, 70, (200, 100, 30, 15)),      # Low — orange
            (70, 180, (30, 100, 30, 10)),      # In range — green
            (180, 250, (200, 160, 30, 12)),    # High — yellow
            (250, 400, (180, 30, 30, 15)),     # Very high — red
        ]

        for zone_lo, zone_hi, rgba in zones:
            if zone_hi < y_min or zone_lo > y_max:
                continue
            clamped_lo = max(zone_lo, y_min)
            clamped_hi = min(zone_hi, y_max)
            # y is inverted (higher value = higher on screen = lower y)
            py_top = chart.y + h * (1 - (clamped_hi - y_min) / (y_max - y_min))
            py_bot = chart.y + h * (1 - (clamped_lo - y_min) / (y_max - y_min))
            zone_surf = pygame.Surface((chart.width, int(py_bot - py_top)), pygame.SRCALPHA)
            zone_surf.fill(rgba)
            self.buffer.blit(zone_surf, (chart.x, int(py_top)))

    def _draw_grid(self, chart):
        """Draw time grid lines and Y axis labels."""
        start, end, visible_steps = self._visible_range()

        # Determine grid interval based on zoom
        if self.pixels_per_step >= 3:
            interval_steps = 12  # 1 hour
            major_interval = 12 * 6  # 6 hours
        elif self.pixels_per_step >= 1:
            interval_steps = 12 * 3  # 3 hours
            major_interval = 12 * 12  # 12 hours
        else:
            interval_steps = 12 * 6  # 6 hours
            major_interval = STEPS_PER_DAY  # 24 hours

        # Vertical grid lines (time)
        first_line = (start // interval_steps) * interval_steps
        for step in range(first_line, end + 1, interval_steps):
            px = self._step_to_x(step, chart)
            if px < chart.x or px > chart.x + chart.width:
                continue
            is_major = (step % major_interval) == 0
            is_day = (step % STEPS_PER_DAY) == 0
            color = TEXT_DIM if is_day else (GRID_COLOR_MAJOR if is_major else GRID_COLOR)
            width = 2 if is_day else 1
            pygame.draw.line(self.buffer, color, (int(px), chart.y), (int(px), chart.y + chart.height), width)

            # Time label
            label = format_time(step)
            if is_day:
                day_num = step // STEPS_PER_DAY + 1
                label = f"Day {day_num}"
            draw_text(self.buffer, self.font_sm, label,
                      int(px) + 3, chart.y + chart.height + 2, TEXT_DIM)

        # Y axis for visible curves — draw on right side of each curve's area
        # We'll draw Y labels on the far right
        active_curves = [(i, c) for i, c in enumerate(CURVES) if self.curve_visible[i]]
        if active_curves:
            # Use the first visible curve for Y axis on the left
            for ci, (idx, curve_def) in enumerate(active_curves):
                y_min, y_max = curve_def['y_min'], curve_def['y_max']
                label_x = chart.x + chart.width + 5 + ci * 65

                if label_x + 60 > self.win_w:
                    break

                # Y axis ticks
                n_ticks = 5
                for ti in range(n_ticks + 1):
                    frac = ti / n_ticks
                    val = y_min + (y_max - y_min) * frac
                    py = chart.y + chart.height * (1 - frac)

                    # Tick line
                    if ci == 0:
                        pygame.draw.line(self.buffer, GRID_COLOR,
                                         (chart.x, int(py)), (chart.x + chart.width, int(py)))

                    # Label
                    if val == int(val):
                        txt = str(int(val))
                    else:
                        txt = f"{val:.1f}"
                    draw_text(self.buffer, self.font_sm, txt, label_x, int(py) - 6, curve_def['color'])

                # Curve name at top
                draw_text(self.buffer, self.font_sm, curve_def['name'][:8],
                          label_x, chart.y - 14, curve_def['color'])

    def _draw_curves(self, chart):
        """Draw all visible curves."""
        start, end, _ = self._visible_range()
        if end <= start or self.data is None:
            return

        for i, curve_def in enumerate(CURVES):
            if not self.curve_visible[i]:
                continue

            key = curve_def['key']
            y_min, y_max = curve_def['y_min'], curve_def['y_max']
            color = curve_def['color']
            data = self.data[key]

            # Build point list
            points = []
            for step in range(start, min(end, len(data))):
                px = self._step_to_x(step, chart)
                val = data[step]
                frac = (val - y_min) / (y_max - y_min) if y_max != y_min else 0.5
                frac = max(0, min(1, frac))
                py = chart.y + chart.height * (1 - frac)
                points.append((px, py))

            if len(points) >= 2:
                # Draw with anti-aliased lines
                pygame.draw.lines(self.buffer, color, False, points, 2)

            # For BG curve, also draw fill below certain thresholds
            if key == 'bg_observed' and len(points) >= 2:
                # Highlight lows
                for j in range(len(points) - 1):
                    step_j = start + j
                    if step_j < len(data) and data[step_j] < 70:
                        px = points[j][0]
                        py = points[j][1]
                        pygame.draw.circle(self.buffer, (255, 60, 60), (int(px), int(py)), 3)
                    elif step_j < len(data) and data[step_j] > 300:
                        px = points[j][0]
                        py = points[j][1]
                        pygame.draw.circle(self.buffer, (255, 200, 40), (int(px), int(py)), 2)

    def _draw_crosshair(self, chart):
        """Draw crosshair and tooltip at mouse position."""
        mx, my = pygame.mouse.get_pos()
        if not chart.collidepoint(mx, my):
            self.hovered_step = None
            return

        # Find step under cursor
        step = int(self.scroll_x + (mx - chart.x) / self.pixels_per_step)
        if step < 0 or step >= self.total_steps:
            self.hovered_step = None
            return

        self.hovered_step = step

        # Vertical line
        pygame.draw.line(self.buffer, (80, 80, 100), (mx, chart.y), (mx, chart.y + chart.height), 1)

        # Tooltip
        tooltip_lines = [format_day_time(step)]
        for i, curve_def in enumerate(CURVES):
            if not self.curve_visible[i]:
                continue
            key = curve_def['key']
            if step < len(self.data[key]):
                val = self.data[key][step]
                tooltip_lines.append(f"{curve_def['name']}: {val:.1f} {curve_def['unit']}")

        # Sick/rare indicators
        if step < len(self.data['is_sick']) and self.data['is_sick'][step]:
            tooltip_lines.append("⚠ SICK")
        if step < len(self.data['is_rare_day']) and self.data['is_rare_day'][step]:
            tooltip_lines.append("⚠ RARE DAY")

        # Draw tooltip box
        tt_w = max(self.font_sm.size(line)[0] for line in tooltip_lines) + 16
        tt_h = len(tooltip_lines) * 16 + 8
        tt_x = min(mx + 15, self.win_w - tt_w - 5)
        tt_y = max(chart.y, my - tt_h // 2)

        tt_surf = pygame.Surface((tt_w, tt_h), pygame.SRCALPHA)
        tt_surf.fill((20, 20, 30, 220))
        self.buffer.blit(tt_surf, (tt_x, tt_y))
        pygame.draw.rect(self.buffer, GRID_COLOR_MAJOR, (tt_x, tt_y, tt_w, tt_h), 1)

        for j, line in enumerate(tooltip_lines):
            color = TEXT_BRIGHT if j == 0 else TEXT_COLOR
            draw_text(self.buffer, self.font_sm, line, tt_x + 8, tt_y + 4 + j * 16, color)

        # Dots on curves at this step
        for i, curve_def in enumerate(CURVES):
            if not self.curve_visible[i]:
                continue
            key = curve_def['key']
            if step < len(self.data[key]):
                val = self.data[key][step]
                y_min, y_max = curve_def['y_min'], curve_def['y_max']
                frac = (val - y_min) / (y_max - y_min) if y_max != y_min else 0.5
                frac = max(0, min(1, frac))
                py = chart.y + chart.height * (1 - frac)
                pygame.draw.circle(self.buffer, curve_def['color'], (mx, int(py)), 5)
                pygame.draw.circle(self.buffer, (255, 255, 255), (mx, int(py)), 5, 1)

    def _draw_header(self):
        """Draw header bar."""
        pygame.draw.rect(self.buffer, PANEL_COLOR, (SIDEBAR_WIDTH, 0, self.win_w - SIDEBAR_WIDTH, HEADER_HEIGHT))
        pygame.draw.line(self.buffer, GRID_COLOR_MAJOR,
                         (SIDEBAR_WIDTH, HEADER_HEIGHT), (self.win_w, HEADER_HEIGHT))

        # Current time info
        if self.total_steps > 0:
            total_hours = self.total_steps * DT_MINUTES / 60
            total_days = total_hours / 24
            draw_text(self.buffer, self.font_lg,
                      f"Generated: {total_hours:.0f}h ({total_days:.1f} days)  |  "
                      f"Steps: {self.total_steps}  |  "
                      f"Zoom: {self.pixels_per_step:.1f}px/step",
                      SIDEBAR_WIDTH + 15, 15, TEXT_COLOR)

    def _draw_footer(self):
        """Draw footer bar."""
        footer_y = self.win_h - FOOTER_HEIGHT
        pygame.draw.rect(self.buffer, PANEL_COLOR, (0, footer_y, self.win_w, FOOTER_HEIGHT))

        # Scrollbar
        chart = self._chart_rect()
        if self.total_steps > 0:
            visible_frac = min(1.0, chart.width / (self.total_steps * self.pixels_per_step))
            scroll_frac = self.scroll_x / max(1, self.total_steps)
            sb_x = SIDEBAR_WIDTH + int(scroll_frac * (self.win_w - SIDEBAR_WIDTH - 20))
            sb_w = max(20, int(visible_frac * (self.win_w - SIDEBAR_WIDTH - 20)))
            pygame.draw.rect(self.buffer, GRID_COLOR,
                             (SIDEBAR_WIDTH + 5, footer_y + 8, self.win_w - SIDEBAR_WIDTH - 10, 14))
            pygame.draw.rect(self.buffer, ACCENT, (sb_x, footer_y + 8, sb_w, 14))

    def run(self):
        """Main loop."""
        running = True
        scroll_speed = 20
        
        # ADDED: Flag to track if the screen actually needs to be updated
        needs_redraw = True 

        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                # Any event (mouse move, click, keypress) means we should redraw
                needs_redraw = True 
                
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self.win_w = max(MIN_WINDOW_W, event.w)
                    self.win_h = max(MIN_WINDOW_H, event.h)
                    self.screen = pygame.display.set_mode(
                        (self.win_w, self.win_h),
                        pygame.RESIZABLE | pygame.DOUBLEBUF
                    )
                    self.buffer = pygame.Surface((self.win_w, self.win_h))

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_SPACE:
                        self._generate(24)
                        chart = self._chart_rect()
                        visible_steps = int(chart.width / self.pixels_per_step)
                        self.scroll_x = max(0, self.total_steps - visible_steps)

                    elif event.key == pygame.K_r:
                        self._reseed(np.random.randint(0, 100000))

                    elif event.key in range(pygame.K_0, pygame.K_9 + 1):
                        digit = event.key - pygame.K_0
                        if digit >= 1 and digit <= 6:
                            self.curve_visible[digit - 1] = not self.curve_visible[digit - 1]
                        elif digit == 0:
                            self._reseed(0)

                    elif event.key == pygame.K_a:
                        all_on = all(self.curve_visible)
                        self.curve_visible = [not all_on] * len(self.curve_visible)

                    elif event.key == pygame.K_HOME:
                        self.scroll_x = 0
                    elif event.key == pygame.K_END:
                        chart = self._chart_rect()
                        visible_steps = int(chart.width / self.pixels_per_step)
                        self.scroll_x = max(0, self.total_steps - visible_steps)

                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.pixels_per_step = min(20, self.pixels_per_step * 1.3)
                    elif event.key == pygame.K_MINUS:
                        self.pixels_per_step = max(0.1, self.pixels_per_step / 1.3)

                    elif event.key == pygame.K_s:
                        fname = f"t1dm_seed{self.seed}_{int(time.time())}.png"
                        pygame.image.save(self.buffer, fname)
                        print(f"Screenshot saved: {fname}")

                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_x -= event.y * scroll_speed
                    self.scroll_x = max(0, min(self.total_steps - 10, self.scroll_x))

            # Keyboard scrolling (continuous)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.scroll_x = max(0, self.scroll_x - scroll_speed)
                needs_redraw = True # Trigger redraw while holding key
            if keys[pygame.K_RIGHT]:
                self.scroll_x = min(max(0, self.total_steps - 10), self.scroll_x + scroll_speed)
                needs_redraw = True # Trigger redraw while holding key

            # === ONLY DRAW IF SOMETHING CHANGED ===
            if needs_redraw:
                self.buffer.fill(BG_COLOR)

                chart = self._chart_rect()

                self._draw_bg_zones(chart)
                self._draw_grid(chart)

                # Chart border
                pygame.draw.rect(self.buffer, GRID_COLOR_MAJOR, chart, 1)

                self._draw_curves(chart)
                self._draw_crosshair(chart)
                self._draw_sidebar()
                self._draw_header()
                self._draw_footer()

                self.screen.blit(self.buffer, (0, 0))
                pygame.display.flip()
                
                # Reset flag so we don't draw next frame unless needed
                needs_redraw = False 

        pygame.quit()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='T1DM Simulator Visualizer')
    parser.add_argument('--seed', type=int, default=42, help='Initial seed')
    parser.add_argument('--bg', type=float, default=None, help='Initial blood glucose')
    parser.add_argument('--hours', type=float, default=24, help='Initial hours to generate')
    args = parser.parse_args()

    viz = Visualizer()
    if args.seed != 42:
        viz._reseed(args.seed)
    if args.bg is not None:
        viz.sim.state.bg = args.bg
    if args.hours != 24:
        viz._generate(args.hours - 24)

    viz.run()
