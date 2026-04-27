from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from manimlib import *

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.ship_roll_setup import build_experiment, run_controller  # noqa: E402


class HybridAdaptiveIdea(Scene):
    def construct(self):
        self.camera.background_color = "#05070b"

        experiment = build_experiment()
        result = run_controller(experiment, "hybrid_adaptive")
        phase = PhaseMapper(center=3.82 * LEFT + 0.52 * DOWN, scale=1.95)

        title = Text("Hybrid adaptive ship-roll stabilization", font_size=31)
        title.to_edge(UP)
        title.set_color(WHITE)
        title.set_backstroke(width=7)

        subtitle = Text(
            "outer Lyapunov-PDE entrance  |  one-time switch  |  local adaptive stabilizer",
            font_size=19,
        )
        subtitle.next_to(title, DOWN, buff=0.16)
        subtitle.set_color(GREY_B)
        subtitle.set_backstroke(width=6)

        phase_plane = self.make_phase_plane(experiment, result, phase)
        explanation = self.make_explanation_box()
        wave_panel = self.make_wave_panel(result)

        self.play(FadeIn(title, shift=0.25 * UP), FadeIn(subtitle, shift=0.15 * UP), run_time=1.1)
        self.play(*[ShowCreation(mob) for mob in phase_plane["base"]], run_time=1.6)
        self.play(FadeIn(explanation, shift=0.25 * RIGHT), run_time=1.0)

        self.play(
            *[GrowArrow(arrow) for arrow in phase_plane["outer_arrows"]],
            FadeIn(phase_plane["outer_label"], shift=0.15 * UP),
            run_time=1.5,
        )
        self.wait(0.3)

        traj_path = phase_plane["trajectory"]
        ship_dot = Dot(traj_path.get_start(), radius=0.075, color=YELLOW)
        halo = Circle(radius=0.18, color=YELLOW).move_to(ship_dot)
        halo.set_stroke(YELLOW, 2, opacity=0.65)
        switch_marker = phase_plane["switch_marker"]
        switch_label = phase_plane["switch_label"]

        self.add(ship_dot, halo)
        self.play(
            ShowCreation(traj_path),
            MoveAlongPath(ship_dot, traj_path),
            MoveAlongPath(halo, traj_path),
            run_time=5.0,
            rate_func=smooth,
        )
        self.play(FadeIn(switch_marker, scale=0.8), FadeIn(switch_label, shift=0.2 * UP), run_time=0.8)
        self.wait(0.3)

        self.play(FadeIn(wave_panel, shift=0.30 * LEFT), run_time=1.1)
        self.play(ShowCreation(wave_panel.true_curve), run_time=1.3)
        self.play(ShowCreation(wave_panel.estimate_curve), run_time=2.1)
        self.play(
            ShowCreation(wave_panel.error_curve),
            FadeIn(wave_panel.convergence_note, shift=0.15 * UP),
            run_time=1.3,
        )

        local_glow = phase_plane["local_glow"]
        self.play(
            FadeIn(local_glow, scale=1.08),
            FadeIn(phase_plane["local_label"], shift=0.15 * DOWN),
            run_time=1.1,
        )

        conclusion = Text(
            "Fourier adaptation learns d(t); the switch only certifies when the local adaptive law is used.",
            font_size=18,
        )
        conclusion.to_edge(DOWN, buff=0.22)
        conclusion.set_color(WHITE)
        conclusion.set_backstroke(width=7)
        self.play(FadeIn(conclusion, shift=0.2 * UP), run_time=0.9)
        self.wait(2.0)

    def make_phase_plane(self, experiment, result, phase):
        P = experiment.lyapunov.P
        levels = experiment.levels
        valid = np.isfinite(result.x[:, 0]) & np.isfinite(result.x[:, 1])
        x_hist = result.x[valid]
        t_hist = result.t[valid]
        idx = np.linspace(0, len(x_hist) - 1, 320).astype(int)
        traj_points = [phase.map_state(x_hist[i]) for i in idx]
        trajectory = VMobject()
        trajectory.set_points_smoothly(traj_points)
        trajectory.set_stroke(color=YELLOW, width=4.5, opacity=0.95)

        axes = VGroup(
            Line(phase.map_xy(-1.05, 0.0), phase.map_xy(1.15, 0.0)),
            Line(phase.map_xy(0.0, -1.25), phase.map_xy(0.0, 1.25)),
        )
        axes.set_stroke(GREY_B, width=2.0, opacity=0.9)
        theta_label = Text("theta", font_size=23).next_to(axes[0], RIGHT, buff=0.10)
        p_label = Text("p", font_size=23).next_to(axes[1], UP, buff=0.10)
        for label in [theta_label, p_label]:
            label.set_color(GREY_B)
            label.set_backstroke(width=5)

        r_curve = self.ellipse(P, levels.r_gamma, phase, color=BLUE_C, width=4.0)
        abar_curve = self.ellipse(P, levels.a_bar, phase, color=ORANGE, width=3.0)
        aloc_curve = self.ellipse(P, levels.a_loc, phase, color=GREEN_C, width=3.0)
        r_curve = DashedVMobject(r_curve, num_dashes=54)
        abar_curve = DashedVMobject(abar_curve, num_dashes=48)
        aloc_curve = DashedVMobject(aloc_curve, num_dashes=52)

        local_glow = self.ellipse(P, levels.r_gamma, phase, color=BLUE_C, width=18.0)
        local_glow.set_stroke(BLUE_C, width=18.0, opacity=0.16)

        singular_line = DashedVMobject(
            Line(
                phase.map_xy(-0.95, -(P[1, 0] / P[1, 1]) * (-0.95)),
                phase.map_xy(1.15, -(P[1, 0] / P[1, 1]) * 1.15),
            ),
            num_dashes=30,
        )
        singular_line.set_stroke(GREY_A, width=2.0, opacity=0.85)

        start = Dot(phase.map_state(x_hist[0]), radius=0.09, color=GREEN)
        target = VGroup(
            Line(0.10 * UL, 0.10 * DR),
            Line(0.10 * DL, 0.10 * UR),
        ).move_to(phase.map_xy(0.0, 0.0))
        target.set_stroke(WHITE, width=3)
        finish = VGroup(
            Line(0.11 * UL, 0.11 * DR),
            Line(0.11 * DL, 0.11 * UR),
        ).move_to(phase.map_state(x_hist[-1]))
        finish.set_stroke(RED, width=3)

        switch_i = result.switch_index if result.switch_index is not None else int(0.1 * len(x_hist))
        switch_point = phase.map_state(result.x[switch_i])
        switch_marker = VGroup(
            Line(0.12 * UL, 0.12 * DR),
            Line(0.12 * DL, 0.12 * UR),
        ).move_to(switch_point)
        switch_marker.set_stroke(GREEN_C, width=5)
        switch_label = Text("switch at W = r_gamma", font_size=17)
        switch_label.set_color(GREEN_C)
        switch_label.set_backstroke(width=6)
        switch_label.next_to(switch_marker, DL, buff=0.12)

        legend = VGroup(
            self.legend_item(BLUE_C, "W = r_gamma", dashed=True),
            self.legend_item(ORANGE, "W = a_bar", dashed=True),
            self.legend_item(GREEN_C, "W = a_loc", dashed=True),
            self.legend_item(GREY_A, "singular line", dashed=True),
        )
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        legend.move_to(phase.center + 1.90 * LEFT + 2.35 * UP)

        outer_arrows = VGroup()
        for point in [np.array([0.85, 0.00]), np.array([0.55, -0.95]), np.array([-0.35, 0.78])]:
            start = phase.map_state(point)
            end = phase.map_state(0.62 * point)
            outer_arrows.add(Arrow(start, end, buff=0.02, color=TEAL_A, stroke_width=4.0))

        outer_label = Text("outer adaptive Lyapunov-PDE field", font_size=18)
        outer_label.set_color(TEAL_A)
        outer_label.set_backstroke(width=6)
        outer_label.move_to(phase.center + 0.10 * RIGHT + 2.62 * UP)

        local_label = Text("inside: local adaptive stabilizer", font_size=18)
        local_label.set_color(BLUE_B)
        local_label.set_backstroke(width=6)
        local_label.move_to(phase.center + 0.35 * RIGHT + 1.55 * DOWN)

        base = VGroup(
            axes,
            theta_label,
            p_label,
            aloc_curve,
            abar_curve,
            r_curve,
            singular_line,
            start,
            target,
            finish,
            legend,
        )
        group = VGroup(base, outer_arrows, outer_label, trajectory, switch_marker, switch_label, local_glow, local_label)

        return {
            "base": base,
            "group": group,
            "outer_arrows": outer_arrows,
            "outer_label": outer_label,
            "trajectory": trajectory,
            "switch_marker": switch_marker,
            "switch_label": switch_label,
            "local_glow": local_glow,
            "local_label": local_label,
        }

    def ellipse(self, P, level, phase, color, width):
        theta = np.linspace(0.0, 2.0 * np.pi, 360)
        circle = np.vstack((np.cos(theta), np.sin(theta)))
        vals, vecs = np.linalg.eigh(P)
        transform = vecs @ np.diag(1.0 / np.sqrt(vals))
        points = np.sqrt(level) * (transform @ circle)
        mapped = [phase.map_xy(points[0, i], points[1, i]) for i in range(points.shape[1])]
        curve = VMobject()
        curve.set_points_smoothly(mapped)
        curve.close_path()
        curve.set_stroke(color=color, width=width)
        return curve

    def legend_item(self, color, text, dashed=False):
        line = Line(LEFT * 0.24, RIGHT * 0.24)
        line.set_stroke(color, width=3)
        if dashed:
            line = DashedVMobject(line, num_dashes=5)
            line.set_stroke(color, width=3)
        label = Text(text, font_size=19)
        label.set_color(GREY_A)
        label.set_backstroke(width=5)
        return VGroup(line, label).arrange(RIGHT, buff=0.13)

    def make_explanation_box(self):
        lines = VGroup(
            Text("Outside Omega_gamma:", font_size=18, weight=BOLD).set_color(TEAL_A),
            Text("outer adaptive law drives the state inward", font_size=16).set_color(WHITE),
            Text("At W = r_gamma:", font_size=18, weight=BOLD).set_color(GREEN_C),
            Text("switch using a measurable Lyapunov level", font_size=16).set_color(WHITE),
            Text("Inside:", font_size=18, weight=BOLD).set_color(BLUE_B),
            Text("Fourier estimate cancels the wave moment", font_size=16).set_color(WHITE),
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.09)
        box = SurroundingRectangle(lines, color=GREY_B, buff=0.20)
        box.set_fill(BLACK, opacity=0.62)
        box.set_stroke(GREY_B, width=1.5, opacity=0.85)
        group = VGroup(box, lines)
        group.move_to(3.18 * RIGHT + 1.18 * UP)
        for mob in group:
            mob.set_backstroke(width=5)
        return group

    def make_wave_panel(self, result):
        panel_center = 3.20 * RIGHT + 1.08 * DOWN
        panel = PanelMapper(panel_center, width=5.35, height=1.70)
        base = panel.axes("time", "moment")

        t = result.t
        window = t <= 45.0
        t_small = t[window]
        d_true = result.d_true[window]
        d_hat = result.d_hat[window]
        panel.set_range(t_small, d_true, d_hat, y_min=-1.08, y_max=1.08)
        true_curve = panel.curve(t_small, d_true, color=ORANGE, width=3.0)
        estimate_curve = panel.curve(t_small, d_hat, color=GREEN_C, width=3.2)

        error_panel = PanelMapper(panel_center + 1.20 * DOWN, width=5.35, height=0.76)
        error_base = error_panel.axes("time", "error")
        error = np.abs(d_true - d_hat)
        error_panel.set_range(t_small, error, y_min=0.0, y_max=0.60)
        error_curve = error_panel.curve(t_small, error, color=RED_C, width=2.8)

        title = Text("adaptive Fourier approximation of the wave moment", font_size=18)
        title.set_color(WHITE)
        title.set_backstroke(width=5)
        title.next_to(base, UP, buff=0.10)

        legend = VGroup(
            self.legend_item(ORANGE, "true d(t)", dashed=False),
            self.legend_item(GREEN_C, "estimate d_hat(t)", dashed=False),
            self.legend_item(RED_C, "|d(t) - d_hat(t)|", dashed=False),
        )
        legend.arrange(RIGHT, buff=0.24)
        legend.scale(0.76)
        legend.next_to(error_base, DOWN, buff=0.04)

        early_tag = Text("initial mismatch", font_size=14)
        early_tag.set_color(RED_C)
        early_tag.set_backstroke(width=5)
        early_tag.move_to(panel.map_xy(5.0, -0.92))

        convergence_note = Text("after adaptation: d_hat(t) follows d(t)", font_size=15)
        convergence_note.set_color(GREEN_C)
        convergence_note.set_backstroke(width=6)
        convergence_note.move_to(panel.map_xy(31.0, 0.90))

        group = VGroup(base, error_base, title, legend, early_tag)
        group.true_curve = true_curve
        group.estimate_curve = estimate_curve
        group.error_curve = error_curve
        group.convergence_note = convergence_note
        return group

    def make_energy_panel(self, experiment, result):
        panel_center = 3.22 * RIGHT + 2.78 * DOWN
        panel = PanelMapper(panel_center, width=4.20, height=1.02)
        base = panel.axes("time", "energy")
        t = result.t
        window = t <= 6.0
        t_small = t[window]
        w = result.W[window]
        v = result.V_gamma[window]
        panel.set_range(t_small, w, v)

        w_curve = panel.curve(t_small, w, color=BLUE_C, width=3.0)
        v_curve = panel.curve(t_small, v, color=YELLOW, width=3.0)
        r_line = Line(
            panel.map_xy(t_small[0], experiment.levels.r_gamma),
            panel.map_xy(t_small[-1], experiment.levels.r_gamma),
        )
        r_line.set_stroke(BLUE_B, width=2.0, opacity=0.85)

        title = Text("Lyapunov energy drops before the switch", font_size=17)
        title.set_color(WHITE)
        title.set_backstroke(width=5)
        title.next_to(base, UP, buff=0.08)

        group = VGroup(base, r_line, title)
        group.w_curve = w_curve
        group.v_curve = v_curve
        return group


class PhaseMapper:
    def __init__(self, center, scale):
        self.center = center
        self.scale = scale

    def map_xy(self, theta, p):
        return self.center + self.scale * theta * RIGHT + self.scale * p * UP

    def map_state(self, x):
        return self.map_xy(float(x[0]), float(x[1]))


class PanelMapper:
    def __init__(self, center, width, height):
        self.center = center
        self.width = width
        self.height = height

    def axes(self, x_label, y_label):
        box = Rectangle(width=self.width, height=self.height)
        box.move_to(self.center)
        box.set_fill(BLACK, opacity=0.42)
        box.set_stroke(GREY_B, width=1.4, opacity=0.8)
        x_axis = Line(self.center + 0.45 * self.width * LEFT + 0.37 * self.height * DOWN,
                      self.center + 0.45 * self.width * RIGHT + 0.37 * self.height * DOWN)
        y_axis = Line(self.center + 0.45 * self.width * LEFT + 0.37 * self.height * DOWN,
                      self.center + 0.45 * self.width * LEFT + 0.37 * self.height * UP)
        axes = VGroup(box, x_axis, y_axis)
        axes.set_stroke(GREY_B, width=1.4)
        return axes

    def map_xy(self, t, y):
        if not hasattr(self, "t_min"):
            raise RuntimeError("PanelMapper.set_range() must be called before mapping points.")
        x_norm = (float(t) - self.t_min) / (self.t_max - self.t_min + 1e-12)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = (float(y) - self.y_min) / (self.y_max - self.y_min + 1e-12)
        x = -0.42 * self.width + 0.84 * self.width * x_norm
        yv = -0.28 * self.height + 0.56 * self.height * np.clip(y_norm, 0.0, 1.0)
        return self.center + x * RIGHT + yv * UP

    def set_range(self, t, *series, y_min=None, y_max=None):
        t = np.asarray(t, dtype=float)
        self.t_min = float(np.min(t))
        self.t_max = float(np.max(t))
        if y_min is None or y_max is None:
            values = np.concatenate([np.asarray(y, dtype=float).ravel() for y in series])
            values = values[np.isfinite(values)]
            if len(values) == 0:
                values = np.array([0.0, 1.0])
            spread = max(1e-6, float(np.max(values) - np.min(values)))
            if y_min is None:
                y_min = float(np.min(values) - 0.12 * spread)
            if y_max is None:
                y_max = float(np.max(values) + 0.12 * spread)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def curve(self, t, y, color, width):
        if not hasattr(self, "t_min"):
            self.set_range(t, y)
        sample = np.linspace(0, len(t) - 1, 220).astype(int)
        points = [self.map_xy(t[i], y[i]) for i in sample]
        curve = VMobject()
        curve.set_points_smoothly(points)
        curve.set_stroke(color=color, width=width)
        return curve
