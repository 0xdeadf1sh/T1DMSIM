"""
T1DM Simulator — Interactive Visualizer (Pygame)
==================================================
Controls:
  SPACE       — Generate next 24 hours
  R           — Reseed with random seed
  LEFT/RIGHT  — Scroll timeline
  HOME        — Jump to start
  END         — Jump to end
  +/-         — Zoom in/out on time axis
  1-9, 0      — Toggle individual curve visibility
  A           — Toggle all curves on/off
  F           — Cycle text size (small / medium / large)
  S           — Screenshot (saves PNG)
  Q / ESC     — Quit

Curves (toggle with number keys):
  1 — Blood Glucose (observed)
  2 — Carb intake curve
  3 — Insulin (total)
  4 — Basal insulin
  5 — Bolus insulin
  6 — Insulin Resistance (multiplier; >1 = resistant, <1 = sensitive)
  7 — Exercise curve
  8 — BG Delta
  9 — Hepatic Glucose Output
  0 — Glucose In
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

from simulator import (T1DMSimulator, DT_MINUTES, SIMULATION_START_DAY_OF_WEEK,
                       BG_CLAMP_MIN, BG_CLAMP_MAX, SIMULATOR_WARMUP_HOURS)

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
WARMUP_HOURS = int(SIMULATOR_WARMUP_HOURS)  # discarded before the displayed window starts; never a local literal, or the GUI shows a settling transient the corpus never contains
WARMUP_DAYS = WARMUP_HOURS // 24
DISPLAY_START_DOW = (SIMULATION_START_DAY_OF_WEEK + WARMUP_DAYS) % 7

# Nocturnal hours shaded in the chart background (22:00-06:00)
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
NIGHT_OVERLAY_RGBA = (4, 26, 22, 50)   # Cold teal wash — the Grid at night

DEFAULT_ZOOM_HOURS = 6.0  # Initial visible window width

# Transient "screenshot saved" modal: total dwell, plus the trailing fade window.
SCREENSHOT_TOAST_SECONDS = 3.0
SCREENSHOT_TOAST_FADE = 0.7

# ============================================================================
# VISUAL THEME — "Tron Legacy" (lime neon on a black Grid)
# ============================================================================
# Deep black backdrop, electric-lime hero accent, cyan/orange/magenta neon
# secondaries. Curves are drawn with a glow underlay (see draw_glow_lines) so
# the bright core reads as luminous against the dark field.

# Colors
BG_COLOR        = (3, 6, 5)          # Near-black with a faint green cast
PANEL_COLOR     = (6, 12, 11)        # Panels barely lift off the void
GRID_COLOR      = (18, 46, 30)       # Dim lime Grid lines
GRID_COLOR_MAJOR= (34, 84, 52)       # Brighter lime Grid lines
TEXT_COLOR       = (170, 230, 180)   # Pale lime
TEXT_DIM         = (78, 128, 92)     # Muted lime
TEXT_BRIGHT      = (215, 255, 200)   # Hot lime-white
ACCENT           = (170, 255, 60)    # Electric lime — the signature glow

# Curve colors — neon spectrum tuned for a black field
COLOR_BG_TRACE   = (170, 255, 60)    # Electric lime for BG (the hero trace)
COLOR_BG_OBS     = (170, 255, 60)    # Electric lime for CGM reading
COLOR_CARB       = (255, 150, 20)    # Neon amber for carbs (Tron contrast)
COLOR_INSULIN    = (40, 210, 255)    # Neon cyan for insulin
COLOR_IS         = (200, 100, 255)   # Neon magenta for IS
COLOR_EXERCISE   = (255, 55, 130)    # Hot pink for exercise
COLOR_DELTA      = (90, 255, 220)    # Aqua-lime for delta

# BG zone colors (for background shading)
COLOR_LOW        = (180, 40, 40, 25)
COLOR_IN_RANGE   = (40, 120, 40, 15)
COLOR_HIGH       = (180, 140, 20, 20)
COLOR_VERY_HIGH  = (180, 40, 40, 20)

# Layout (base values — scaled at runtime by the current text scale)
BASE_SIDEBAR_WIDTH = 320
BASE_HEADER_HEIGHT = 56
BASE_FOOTER_HEIGHT = 34
CHART_PADDING    = 10
MIN_WINDOW_W     = 1200
MIN_WINDOW_H     = 700

# Base font sizes (medium = 1.0× of these)
BASE_FONT_SM = 15
BASE_FONT_MD = 18
BASE_FONT_LG = 24
BASE_FONT_XL = 30

# Text scale presets — cycled at runtime with the F key.
# small ≈ pre-feature defaults; medium is baseline; large (the default) suits
# hi-DPI / presentation and the maximized-on-open window.
TEXT_SCALES = {
    'small':  0.80,
    'medium': 1.00,
    'large':  1.25,
}
TEXT_SCALE_ORDER = ['small', 'medium', 'large']
DEFAULT_TEXT_SCALE = 'large'

STEPS_PER_DAY = 24 * 60 // DT_MINUTES  # 288


# ============================================================================
# CURVE DEFINITIONS
# ============================================================================

CURVES = [
    {'key': 'bg_observed', 'name': 'Blood Glucose',     'color': COLOR_BG_OBS,  'unit': 'mg/dL', 'y_min': 20,  'y_max': 500, 'toggle_key': pygame.K_1},
    {'key': 'total_carb',  'name': 'Carb Intake',       'color': COLOR_CARB,    'unit': 'g/step','y_min': 0,   'y_max': 20,   'toggle_key': pygame.K_2},
    {'key': 'total_insulin','name': 'Insulin (total)',  'color': COLOR_INSULIN, 'unit': 'U/step','y_min': 0,   'y_max': 2,   'toggle_key': pygame.K_3},
    {'key': 'basal_insulin','name': 'Basal',             'color': (120, 235, 255), 'unit': 'U/step','y_min': 0, 'y_max': 0.3, 'toggle_key': pygame.K_4},
    {'key': 'bolus_insulin','name': 'Bolus',             'color': (60, 180, 255),  'unit': 'U/step','y_min': 0, 'y_max': 2,   'toggle_key': pygame.K_5},
    {'key': 'insulin_resistance','name': 'Insulin Resistance','color': COLOR_IS,'unit': '×',  'y_min': 0,   'y_max': 3,   'toggle_key': pygame.K_6},
    {'key': 'total_exercise','name': 'Exercise',        'color': COLOR_EXERCISE,'unit': 'g/step','y_min': 0,   'y_max': 10,   'toggle_key': pygame.K_7},
    {'key': 'bg_delta',    'name': 'BG Delta',          'color': COLOR_DELTA,   'unit': 'mg/dL', 'y_min': -20, 'y_max': 10,  'toggle_key': pygame.K_8},
    {'key': 'hgo',          'name': 'Hepatic Output',   'color': (230, 240, 70), 'unit': 'g/step', 'y_min': 0, 'y_max': 1.5, 'toggle_key': pygame.K_9},
    {'key': 'glucose_in', 'name': 'Glucose In',         'color': (255, 90, 90), 'unit': 'g/step', 'y_min': 0, 'y_max': 20, 'toggle_key': pygame.K_0},
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


def _dim(color, factor):
    """Scale an RGB color toward black by `factor` (0..1)."""
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def draw_glow_lines(surface, color, points, width=2, glow=True):
    """Tron-style polyline: a bright core wrapped in dimmer, wider halos.

    On a near-black field a halo painted in progressively darker shades of the
    line's own hue reads as a neon glow — brighter than the void at every ring,
    brightest at the core. Cheap (no per-pixel alpha) and redraw is event-gated.
    """
    if len(points) < 2:
        return
    if glow:
        pygame.draw.lines(surface, _dim(color, 0.22), False, points, width + 6)
        pygame.draw.lines(surface, _dim(color, 0.45), False, points, width + 2)
    pygame.draw.lines(surface, color, False, points, width)


def draw_glow_rect(surface, color, rect, glow=True):
    """Tron-style rectangle outline with a dim halo around a bright edge."""
    if glow:
        pygame.draw.rect(surface, _dim(color, 0.30), rect, 3)
    pygame.draw.rect(surface, color, rect, 1)


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

        # Open maximized. The WM-managed maximize (via SDL2) respects panels and
        # emits a VIDEORESIZE the main loop consumes to resync win_w/win_h/buffer.
        # Guarded so headless/dummy-driver runs (and any build lacking _sdl2)
        # silently keep the near-full-screen size above.
        try:
            from pygame._sdl2.video import Window
            Window.from_display_module().maximize()
        except Exception:
            pass

        # Off-screen buffer to eliminate flickering
        self.buffer = pygame.Surface((self.win_w, self.win_h))

        # Fonts + scale-dependent layout (sidebar/header/footer dims live on self
        # so they can react to text-scale changes; see _apply_text_scale).
        self.text_scale = DEFAULT_TEXT_SCALE
        self._apply_text_scale()

        # Simulator
        self.seed = 42
        self.sim = T1DMSimulator(seed=self.seed)
        self.data = None
        self.total_steps = 0

        # View state
        self.scroll_x = 0           # Leftmost visible step index
        # Initial zoom: fit DEFAULT_ZOOM_HOURS into the available chart width
        steps_per_hour = 60 // DT_MINUTES
        self.pixels_per_step = self._chart_rect().width / (DEFAULT_ZOOM_HOURS * steps_per_hour)
        # One visibility flag per entry in CURVES, in the same order. Defaults:
        # BG (1), carbs (2), total insulin (3), IR (6), hepatic output (9)
        # visible; the rest hidden but toggleable via the digit keys bound in
        # each CURVES entry's `toggle_key` field.
        self.curve_visible = [True, True, True, False, False, True, False, False, True, False]
        self.hovered_step = None     # Step under mouse cursor

        # Transient "screenshot saved" modal: path string + epoch deadline.
        self.screenshot_msg = None
        self.screenshot_msg_until = 0.0

        # Burn off the first day so display starts after dynamics settle, then
        # generate the initial 24h of visible data.
        self._warmup(WARMUP_HOURS)
        self._generate(24)

        # Clock
        self.clock = pygame.time.Clock()

    def _apply_text_scale(self):
        """Rebuild fonts and scale-dependent layout from self.text_scale."""
        mult = TEXT_SCALES[self.text_scale]
        self._text_scale_mult = mult
        self.font_sm = pygame.font.SysFont("DejaVu Sans Mono", max(8, int(BASE_FONT_SM * mult)))
        self.font_md = pygame.font.SysFont("DejaVu Sans Mono", max(8, int(BASE_FONT_MD * mult)))
        self.font_lg = pygame.font.SysFont("DejaVu Sans Mono", max(8, int(BASE_FONT_LG * mult)))
        self.font_xl = pygame.font.SysFont("DejaVu Sans Mono", max(8, int(BASE_FONT_XL * mult)))
        self.sidebar_width = int(BASE_SIDEBAR_WIDTH * mult)
        self.header_height = int(BASE_HEADER_HEIGHT * mult)
        # Footer must hold the scrollbar; chart leaves room for X-axis time labels above
        # the footer and curve-name labels above the chart, so they never collide.
        self.footer_height = max(int(BASE_FOOTER_HEIGHT * mult), self.font_sm.get_linesize() + int(18 * mult))
        self.time_label_height = self.font_sm.get_linesize() * 2 + 6  # hour row + day row
        self.curve_label_height = self.font_sm.get_linesize() + 4

    def _s(self, n):
        """Scale a layout pixel value by the current text scale."""
        return int(n * self._text_scale_mult)

    def _cycle_text_scale(self):
        idx = TEXT_SCALE_ORDER.index(self.text_scale)
        self.text_scale = TEXT_SCALE_ORDER[(idx + 1) % len(TEXT_SCALE_ORDER)]
        self._apply_text_scale()

    def _generate(self, hours):
        """Generate more data."""
        new_data = self.sim.generate_hours(hours)
        if self.data is None:
            self.data = {k: np.array(v) for k, v in new_data.items()}
        else:
            self.data = {k: np.concatenate([self.data[k], new_data[k]]) for k in self.data}
        self.total_steps = len(self.data['bg'])

    def _warmup(self, hours):
        """Advance the simulator without recording — discards transient startup behavior."""
        self.sim.generate_hours(hours)

    def _reseed(self, seed):
        """Reset with new seed."""
        self.seed = seed
        self.sim = T1DMSimulator(seed=seed)
        self.data = None
        self.total_steps = 0
        self.scroll_x = 0
        self._warmup(WARMUP_HOURS)
        self._generate(24)

    def _chart_rect(self):
        """Get the chart drawing area."""
        x = self.sidebar_width + CHART_PADDING
        y = self.header_height + CHART_PADDING + self.curve_label_height
        w = self.win_w - self.sidebar_width - CHART_PADDING * 2
        h = (self.win_h - self.header_height - self.footer_height
             - self.time_label_height - self.curve_label_height
             - CHART_PADDING * 2)
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
        sidebar = pygame.Rect(0, 0, self.sidebar_width, self.win_h)
        pygame.draw.rect(self.buffer, PANEL_COLOR, sidebar)
        pygame.draw.line(self.buffer, _dim(ACCENT, 0.35),
                         (self.sidebar_width - 2, 0), (self.sidebar_width - 2, self.win_h), 3)
        pygame.draw.line(self.buffer, ACCENT,
                         (self.sidebar_width - 1, 0), (self.sidebar_width - 1, self.win_h))

        line_sm = self.font_sm.get_linesize()
        line_md = self.font_md.get_linesize()

        x = self._s(12)
        y = self._s(12)
        draw_text(self.buffer, self.font_lg, "T1DM Simulator", x, y, ACCENT)
        y += self.font_lg.get_linesize() + self._s(6)

        # Seed + text scale
        draw_text(self.buffer, self.font_md, f"Seed: {self.seed}", x, y, ACCENT)
        draw_text(self.buffer, self.font_sm, f"Text: {self.text_scale}",
                  self.sidebar_width - self._s(12), y, TEXT_DIM, anchor='topright')
        y += line_md + self._s(6)

        # Patient profile
        draw_text(self.buffer, self.font_md, "— Patient Profile —", x, y, TEXT_DIM)
        y += line_md + self._s(4)

        p = self.sim.patient
        profile_items = [
            ("Diet Discipline", p.dietary_discipline, COLOR_CARB),
            ("Attentiveness",   p.attentiveness,      COLOR_INSULIN),
            ("Dose Competence", p.dosing_competence,   COLOR_IS),
            ("Consistency",     p.lifestyle_consistency, COLOR_EXERCISE),
        ]
        bar_w = self._s(100)
        bar_h = self._s(10)
        for label, val, color in profile_items:
            draw_text(self.buffer, self.font_sm, label, x, y, TEXT_DIM)
            bar_x = x + self._s(140)
            bar_y_c = y + (line_sm - bar_h) // 2
            pygame.draw.rect(self.buffer, GRID_COLOR, (bar_x, bar_y_c, bar_w, bar_h))
            fill_w = int(val * bar_w)
            pygame.draw.rect(self.buffer, color, (bar_x, bar_y_c, fill_w, bar_h))
            draw_text(self.buffer, self.font_sm, f"{val:.2f}", bar_x + bar_w + self._s(5), y, TEXT_DIM)
            y += line_sm + self._s(2)

        y += self._s(10)
        draw_text(self.buffer, self.font_md, "— Parameters —", x, y, TEXT_DIM)
        y += line_md + self._s(4)

        summary = self.sim.get_patient_summary()
        param_keys = ['is_base', 'icr', 'correction_factor', 'basal_dose',
                      'basal_duration', 'cgm_check_interval', 'patience_time',
                      'exercise_prob', 'basal_miss_prob', 'slow_carb_pref',
                      'panic_factor']
        param_labels = ['IS Base', 'ICR', 'Correction Factor', 'Basal Dose',
                        'Basal Duration', 'CGM Check Interval', 'Patience Time',
                        'Exercise Prob', 'Basal Miss Prob', 'Slow Carb Pref',
                        'Panic Factor']

        param_col_x = x + self._s(150)
        for label, key in zip(param_labels, param_keys):
            draw_text(self.buffer, self.font_sm, f"{label}:", x, y, TEXT_DIM)
            draw_text(self.buffer, self.font_sm, summary[key], param_col_x, y, TEXT_COLOR)
            y += line_sm

        # Stats
        if self.data is not None and self.total_steps > 0:
            y += self._s(15)
            draw_text(self.buffer, self.font_md, "— Statistics —", x, y, TEXT_DIM)
            y += line_md + self._s(4)

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

            stats_col_x = x + self._s(140)
            for label, val in stats:
                draw_text(self.buffer, self.font_sm, f"{label}:", x, y, TEXT_DIM)
                color = TEXT_COLOR
                if label == "Time in Range":
                    color = COLOR_BG_TRACE if tir > 70 else (COLOR_CARB if tir > 40 else (200, 60, 60))
                elif label == "Status":
                    color = (255, 80, 80)
                elif label == "Today":
                    color = (255, 200, 50)
                draw_text(self.buffer, self.font_sm, val, stats_col_x, y, color)
                y += line_sm

        # Curve legend / toggles. Each row shows the actual digit key bound
        # to that curve via its `toggle_key` field — not its position in the
        # CURVES list — so the on-screen label matches what the keyboard
        # handler does (otherwise inserting a curve in the middle of CURVES
        # would silently drift labels off of the bindings).
        y += self._s(15)
        draw_text(self.buffer, self.font_md, "— Curves (0-9) —", x, y, TEXT_DIM)
        y += line_md + self._s(4)

        for i, curve in enumerate(CURVES):
            prefix = "●" if self.curve_visible[i] else "○"
            color = curve['color'] if self.curve_visible[i] else TEXT_DIM
            tk = curve.get('toggle_key')
            digit = (tk - pygame.K_0) if tk is not None else None
            label = f"[{digit}]" if digit is not None else "[ ]"
            draw_text(self.buffer, self.font_sm, f"{label} {prefix} {curve['name']}", x, y, color)
            y += line_sm

        # Controls
        y += self._s(15)
        draw_text(self.buffer, self.font_md, "— Controls —", x, y, TEXT_DIM)
        y += line_md + self._s(2)
        controls = [
            "SPACE  Generate +24h",
            "R      Random reseed",
            "0-9    Toggle curves",
            "←→     Scroll time",
            "+−     Zoom",
            "HOME   Jump to start",
            "END    Jump to end",
            "A      Toggle all curves",
            "F      Cycle text size",
            "S      Screenshot",
            "Q/ESC  Quit",
        ]
        for line in controls:
            if y + line_sm > self.win_h - self._s(10):
                break
            draw_text(self.buffer, self.font_sm, line, x, y, TEXT_DIM)
            y += line_sm

    def _draw_nocturnal_zones(self, chart):
        """Shade chart background during nocturnal hours (NIGHT_START_HOUR to NIGHT_END_HOUR)."""
        start, end, _ = self._visible_range()
        if start >= end:
            return
        steps_per_hour = 60 // DT_MINUTES
        # Walk forward across the visible range, marking [night_start, night_end) windows.
        # Night windows wrap around midnight, so emit two segments per day.
        first_day = start // STEPS_PER_DAY
        last_day = end // STEPS_PER_DAY
        for day in range(first_day - 1, last_day + 2):
            day_start = day * STEPS_PER_DAY
            # Two pieces of the night that straddle midnight:
            # 1) NIGHT_START_HOUR through end of this calendar day
            seg1_lo = day_start + NIGHT_START_HOUR * steps_per_hour
            seg1_hi = day_start + STEPS_PER_DAY
            # 2) Start of NEXT calendar day through NIGHT_END_HOUR
            seg2_lo = day_start + STEPS_PER_DAY
            seg2_hi = seg2_lo + NIGHT_END_HOUR * steps_per_hour
            for lo, hi in ((seg1_lo, seg1_hi), (seg2_lo, seg2_hi)):
                vis_lo = max(lo, start)
                vis_hi = min(hi, end)
                if vis_hi <= vis_lo:
                    continue
                px_lo = int(self._step_to_x(vis_lo, chart))
                px_hi = int(self._step_to_x(vis_hi, chart))
                if px_hi <= px_lo:
                    continue
                seg_surf = pygame.Surface((px_hi - px_lo, chart.height), pygame.SRCALPHA)
                seg_surf.fill(NIGHT_OVERLAY_RGBA)
                self.buffer.blit(seg_surf, (px_lo, chart.y))

    def _draw_bg_zones(self, chart):
        """Draw colored background zones for BG ranges."""
        if not self.curve_visible[0]:
            return

        curve_def = CURVES[0]
        y_min, y_max = curve_def['y_min'], curve_def['y_max']
        h = chart.height

        zones = [
            (30, 54, (200, 30, 40, 22)),       # Very low — red
            (54, 70, (220, 90, 30, 16)),       # Low — amber
            (70, 180, (90, 200, 50, 14)),      # In range — lime (the safe Grid)
            (180, 250, (210, 170, 30, 13)),    # High — yellow
            (250, 400, (200, 30, 40, 16)),     # Very high — red
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

        # Vertical grid lines (time) + a two-row X axis: hour labels on the
        # first row, the wider "Tue (Day N)" day marker on a second row below it
        # (in the accent colour), so the day string never collides with the
        # dense hourly ticks. Hour labels are thinned when the per-tick spacing
        # is narrower than a label.
        label_y = chart.y + chart.height + 2
        day_y = label_y + self.font_sm.get_linesize()
        hour_label_w = self.font_sm.size("00:00")[0] + self._s(8)
        step_px = max(1.0, interval_steps * self.pixels_per_step)
        hour_every = max(1, int(np.ceil(hour_label_w / step_px)))

        first_line = (start // interval_steps) * interval_steps
        for tick, step in enumerate(range(first_line, end + 1, interval_steps)):
            px = self._step_to_x(step, chart)
            if px < chart.x or px > chart.x + chart.width:
                continue
            is_major = (step % major_interval) == 0
            is_day = (step % STEPS_PER_DAY) == 0
            color = TEXT_DIM if is_day else (GRID_COLOR_MAJOR if is_major else GRID_COLOR)
            width = 2 if is_day else 1
            pygame.draw.line(self.buffer, color, (int(px), chart.y), (int(px), chart.y + chart.height), width)

            if is_day or tick % hour_every == 0:
                draw_text(self.buffer, self.font_sm, format_time(step),
                          int(px) + 3, label_y, TEXT_DIM)
            if is_day:
                day_num = step // STEPS_PER_DAY + 1
                dow = (DISPLAY_START_DOW + (step // STEPS_PER_DAY)) % 7
                draw_text(self.buffer, self.font_sm, f"{DAY_NAMES[dow]} (Day {day_num})",
                          int(px) + 3, day_y, ACCENT)

        # Y axis for visible curves — draw on right side of each curve's area
        # We'll draw Y labels on the far right
        active_curves = [(i, c) for i, c in enumerate(CURVES) if self.curve_visible[i]]
        if active_curves:
            # Use the first visible curve for Y axis on the left
            col_w = self._s(65)
            for ci, (idx, curve_def) in enumerate(active_curves):
                y_min, y_max = curve_def['y_min'], curve_def['y_max']
                label_x = chart.x + chart.width + self._s(5) + ci * col_w

                if label_x + self._s(60) > self.win_w:
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

                    # Label — vertically center against the tick
                    if val == int(val):
                        txt = str(int(val))
                    else:
                        txt = f"{val:.1f}"
                    draw_text(self.buffer, self.font_sm, txt,
                              label_x, int(py) - self.font_sm.get_height() // 2,
                              curve_def['color'])

                # Curve name at top
                draw_text(self.buffer, self.font_sm, curve_def['name'][:8],
                          label_x, chart.y - self.font_sm.get_linesize(), curve_def['color'])

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
                # The BG trace is the hero — give it the fullest glow; the rest
                # carry a lighter halo so the chart doesn't smear into mush.
                hero = (key == 'bg_observed')
                draw_glow_lines(self.buffer, color, points, width=2, glow=hero)

            # For BG curve, also draw fill below certain thresholds
            if key == 'bg_observed' and len(points) >= 2:
                # Highlight lows
                for j in range(len(points) - 1):
                    step_j = start + j
                    if step_j < len(data) and data[step_j] < 70:
                        px = points[j][0]
                        py = points[j][1]
                        pygame.draw.circle(self.buffer, (255, 70, 90), (int(px), int(py)), 3)
                    elif step_j < len(data) and data[step_j] > 300:
                        px = points[j][0]
                        py = points[j][1]
                        pygame.draw.circle(self.buffer, (255, 210, 40), (int(px), int(py)), 2)

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

        # Vertical line — dim lime scanline
        pygame.draw.line(self.buffer, (60, 120, 70), (mx, chart.y), (mx, chart.y + chart.height), 1)

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
        tt_line_h = self.font_sm.get_linesize()
        tt_pad_x = self._s(8)
        tt_pad_y = self._s(4)
        tt_w = max(self.font_sm.size(line)[0] for line in tooltip_lines) + tt_pad_x * 2
        tt_h = len(tooltip_lines) * tt_line_h + tt_pad_y * 2
        tt_x = min(mx + self._s(15), self.win_w - tt_w - self._s(5))
        tt_y = max(chart.y, my - tt_h // 2)

        tt_surf = pygame.Surface((tt_w, tt_h), pygame.SRCALPHA)
        tt_surf.fill((4, 14, 12, 230))
        self.buffer.blit(tt_surf, (tt_x, tt_y))
        pygame.draw.rect(self.buffer, ACCENT, (tt_x, tt_y, tt_w, tt_h), 1)

        for j, line in enumerate(tooltip_lines):
            color = TEXT_BRIGHT if j == 0 else TEXT_COLOR
            draw_text(self.buffer, self.font_sm, line, tt_x + tt_pad_x, tt_y + tt_pad_y + j * tt_line_h, color)

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
                pygame.draw.circle(self.buffer, _dim(curve_def['color'], 0.4), (mx, int(py)), 7)
                pygame.draw.circle(self.buffer, curve_def['color'], (mx, int(py)), 5)
                pygame.draw.circle(self.buffer, TEXT_BRIGHT, (mx, int(py)), 5, 1)

    def _draw_header(self):
        """Draw header bar."""
        pygame.draw.rect(self.buffer, PANEL_COLOR, (self.sidebar_width, 0, self.win_w - self.sidebar_width, self.header_height))
        pygame.draw.line(self.buffer, _dim(ACCENT, 0.35),
                         (self.sidebar_width, self.header_height + 1), (self.win_w, self.header_height + 1), 3)
        pygame.draw.line(self.buffer, ACCENT,
                         (self.sidebar_width, self.header_height), (self.win_w, self.header_height))

        # Current time info — vertically center font_lg in header
        if self.total_steps > 0:
            total_hours = self.total_steps * DT_MINUTES / 60
            total_days = total_hours / 24
            text_y = (self.header_height - self.font_lg.get_height()) // 2
            draw_text(self.buffer, self.font_lg,
                      f"Generated: {total_hours:.0f}h ({total_days:.1f} days)  |  "
                      f"Steps: {self.total_steps}  |  "
                      f"Zoom: {self.pixels_per_step:.1f}px/step",
                      self.sidebar_width + self._s(15), text_y, TEXT_COLOR)

    def _draw_footer(self):
        """Draw footer bar."""
        footer_y = self.win_h - self.footer_height
        pygame.draw.rect(self.buffer, PANEL_COLOR, (0, footer_y, self.win_w, self.footer_height))

        # Scrollbar — centered vertically inside the footer, height scales with text
        chart = self._chart_rect()
        if self.total_steps > 0:
            sb_h = max(8, self._s(14))
            sb_y = footer_y + (self.footer_height - sb_h) // 2
            track_pad = self._s(5)
            track_w = self.win_w - self.sidebar_width - track_pad * 2
            visible_frac = min(1.0, chart.width / (self.total_steps * self.pixels_per_step))
            scroll_frac = self.scroll_x / max(1, self.total_steps)
            sb_x = self.sidebar_width + track_pad + int(scroll_frac * track_w)
            sb_w = max(self._s(20), int(visible_frac * track_w))
            pygame.draw.rect(self.buffer, GRID_COLOR,
                             (self.sidebar_width + track_pad, sb_y, track_w, sb_h))
            pygame.draw.rect(self.buffer, ACCENT, (sb_x, sb_y, sb_w, sb_h))

    def _render_scene(self):
        """Paint the full frame to the off-screen buffer (no screen flip)."""
        self.buffer.fill(BG_COLOR)
        chart = self._chart_rect()

        self._draw_nocturnal_zones(chart)
        self._draw_bg_zones(chart)
        self._draw_grid(chart)

        # Chart border — lime-glow frame around the Grid
        draw_glow_rect(self.buffer, ACCENT, chart)

        self._draw_curves(chart)
        self._draw_crosshair(chart)
        self._draw_sidebar()
        self._draw_header()
        self._draw_footer()
        self._draw_screenshot_toast()  # always last — overlays everything

    def _draw_screenshot_toast(self):
        """Centered, self-dismissing 'screenshot saved' modal with a lime frame.

        Kept out of the saved PNG by clearing screenshot_msg before the capture
        re-render (see the K_s handler). Fades over the final SCREENSHOT_TOAST_FADE
        seconds; the run loop keeps redrawing while it is live so it animates and
        then clears without needing user input.
        """
        if self.screenshot_msg is None:
            return
        remaining = self.screenshot_msg_until - time.time()
        if remaining <= 0:
            self.screenshot_msg = None
            return

        title = "✓ SCREENSHOT SAVED"
        path = self.screenshot_msg
        pad_x, pad_y = self._s(20), self._s(16)
        line_gap = self._s(6)
        title_surf = self.font_md.render(title, True, ACCENT)
        path_surf = self.font_sm.render(path, True, TEXT_BRIGHT)

        w = min(self.win_w - self._s(40),
                max(title_surf.get_width(), path_surf.get_width()) + pad_x * 2)
        h = title_surf.get_height() + path_surf.get_height() + line_gap + pad_y * 2

        toast = pygame.Surface((w, h), pygame.SRCALPHA)
        toast.fill((4, 14, 12, 238))
        toast_rect = toast.get_rect()
        pygame.draw.rect(toast, (*_dim(ACCENT, 0.35), 255), toast_rect, self._s(3))
        pygame.draw.rect(toast, (*ACCENT, 255), toast_rect, 1)

        ty = pad_y
        toast.blit(title_surf, ((w - title_surf.get_width()) // 2, ty))
        ty += title_surf.get_height() + line_gap
        toast.blit(path_surf, ((w - path_surf.get_width()) // 2, ty))

        # Trailing fade: scale the whole toast's alpha down uniformly.
        if remaining < SCREENSHOT_TOAST_FADE and SCREENSHOT_TOAST_FADE > 0:
            alpha = max(0, int(255 * remaining / SCREENSHOT_TOAST_FADE))
            toast.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)

        self.buffer.blit(toast, ((self.win_w - w) // 2, (self.win_h - h) // 2))

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
                        # Toggle the CURVES entry whose toggle_key matches.
                        # Keys not bound to any curve fall through (no-op).
                        for i, c in enumerate(CURVES):
                            if c.get('toggle_key') == event.key:
                                self.curve_visible[i] = not self.curve_visible[i]
                                break

                    elif event.key == pygame.K_a:
                        all_on = all(self.curve_visible)
                        self.curve_visible = [not all_on] * len(self.curve_visible)

                    elif event.key == pygame.K_f:
                        self._cycle_text_scale()

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
                        path = os.path.abspath(fname)
                        # Re-render a clean frame with the toast suppressed so the
                        # confirmation modal never lands in the saved PNG (even if
                        # a prior toast is still on screen). convert(24) drops any
                        # per-pixel alpha the buffer picked up from SRCALPHA zone
                        # blits, so the file is always opaque.
                        self.screenshot_msg = None
                        self._render_scene()
                        pygame.image.save(self.buffer.convert(24), fname)
                        print(f"Screenshot saved: {path}")
                        self.screenshot_msg = path
                        self.screenshot_msg_until = time.time() + SCREENSHOT_TOAST_SECONDS

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

            # Keep redrawing while the screenshot modal is live so it fades and
            # then clears on its own, without waiting for the next user event.
            if self.screenshot_msg is not None:
                needs_redraw = True

            # === ONLY DRAW IF SOMETHING CHANGED ===
            if needs_redraw:
                self._render_scene()

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

    def _nonneg_int(s):
        v = int(s)
        if v < 0:
            raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
        return v

    def _positive_float(s):
        v = float(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {v}")
        return v

    def _bg_float(s):
        v = float(s)
        if not (BG_CLAMP_MIN <= v <= BG_CLAMP_MAX):
            raise argparse.ArgumentTypeError(
                f"must be in [{BG_CLAMP_MIN}, {BG_CLAMP_MAX}] mg/dL, got {v}"
            )
        return v

    parser = argparse.ArgumentParser(description='T1DM Simulator Visualizer')
    parser.add_argument('--seed', type=_nonneg_int, default=42, help='Initial seed (>= 0)')
    parser.add_argument('--bg', type=_bg_float, default=None,
                        help=f'Initial blood glucose ({BG_CLAMP_MIN}-{BG_CLAMP_MAX} mg/dL)')
    parser.add_argument('--hours', type=_positive_float, default=24,
                        help='Initial hours to generate (> 0)')
    args = parser.parse_args()

    viz = Visualizer()
    # Re-init with the requested seed/initial-BG so the displayed window starts
    # in the requested state. We discard the auto-generated default-seed window
    # and rebuild from scratch: simulator with seed → 24h warmup → optional BG
    # override → generate the first displayed day.
    if args.seed != 42 or args.bg is not None:
        viz.seed = args.seed
        viz.sim = T1DMSimulator(seed=args.seed)
        viz.data = None
        viz.total_steps = 0
        viz.scroll_x = 0
        viz._warmup(WARMUP_HOURS)
        if args.bg is not None:
            viz.sim.state.bg = args.bg
            viz.sim.state.bg_observed = args.bg
        viz._generate(24)
    if args.hours > 24:
        viz._generate(args.hours - 24)
    elif args.hours < 24:
        # The constructor always generates a 24h window; for a shorter request,
        # rebuild from scratch and generate exactly args.hours (a bare
        # _generate(args.hours - 24) would pass a negative step count).
        viz.sim = T1DMSimulator(seed=viz.seed)
        viz.data = None
        viz.total_steps = 0
        viz.scroll_x = 0
        viz._warmup(WARMUP_HOURS)
        if args.bg is not None:
            viz.sim.state.bg = args.bg
            viz.sim.state.bg_observed = args.bg
        viz._generate(args.hours)

    viz.run()
