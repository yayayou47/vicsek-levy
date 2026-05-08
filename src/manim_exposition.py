"""
Figure-centric exposition of the manuscript:
    "alpha-stable angular noise reshapes the Vicsek flocking transition"

Render with:
    manim -qm manim_exposition.py FullExposition

Each slide is figure + equation with at most a one-line caption.
The vicsek_composite movie is spliced as intro + outro by
`build_final_video.sh`.
"""

from __future__ import annotations

from pathlib import Path

from manim import (
    DOWN, GRAY, LEFT, RIGHT, UP, WHITE,
    FadeIn, FadeOut, Group, ImageMobject, MathTex, Scene, Tex, Text, VGroup,
    Write, config,
)

# Cream-on-charcoal palette inspired by the manuscript figures
config.background_color = "#fdf7e3"
config.frame_height = 8.0
config.frame_width = 14.22  # 16:9


CHARCOAL = "#1f1f1f"
BLUE_C = "#1f4ea1"
ORANGE_C = "#d76f3a"

FIGS = Path(__file__).resolve().parent.parent / "figures"


def styled_text(s, **kw):
    kw.setdefault("color", CHARCOAL)
    kw.setdefault("font", "DejaVu Sans")
    return Text(s, **kw)


def styled_math(s, **kw):
    kw.setdefault("color", CHARCOAL)
    return MathTex(s, **kw)


def fig_image(name, height=6.0):
    img = ImageMobject(str(FIGS / name))
    img.set_height(height)
    return img


# --- Section 1: Title -------------------------------------------------

class TitleScene(Scene):
    def construct(self):
        title = Tex(
            r"$\alpha$-stable angular noise\\reshapes Vicsek flocking",
            color=CHARCOAL, font_size=52,
        )
        author = styled_text(
            "Yaya Youssouf Yaya", font_size=22, color=GRAY,
        ).next_to(title, DOWN, buff=0.7)

        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(author, shift=UP * 0.2), run_time=0.5)
        self.wait(2.0)
        self.play(FadeOut(VGroup(title, author)), run_time=0.8)


# --- Section 2: Noise PDF + alpha-stable equation ---------------------

class QuestionScene(Scene):
    def construct(self):
        img = fig_image("fig_noise_pdf.png", height=5.6)
        img.shift(UP * 0.4)
        self.play(FadeIn(img), run_time=1.0)

        eq = MathTex(
            r"\widehat p_\alpha(k) = \exp\!\big(-|\eta\,k|^\alpha\big),"
            r"\qquad \alpha \in (0,2]",
            color=CHARCOAL, font_size=40,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(eq), run_time=1.4)
        self.wait(2.8)
        self.play(FadeOut(Group(img, eq)), run_time=0.8)


# --- Section 3: Model schematic + zonal rule --------------------------

class ModelScene(Scene):
    def construct(self):
        img = fig_image("fig_model_schematic.png", height=5.0)
        img.shift(LEFT * 3.4)

        update = MathTex(
            r"\theta_i(t{+}1) = \theta_i^\star(t{+}1) + \xi_i(t)",
            color=CHARCOAL, font_size=34,
        ).move_to([3.2, 2.0, 0])

        zonal = MathTex(
            r"\theta_i^\star = \begin{cases}"
            r"\arg\!\big(\!-\!\!\sum_{\mathcal R_i}\!\vec r_{ij}\big)"
            r"& \mathcal R_i \neq \emptyset \\[2pt]"
            r"\arg\!\big(\sum_{\mathcal A_i} e^{i\theta_j}\big)"
            r"& \mathcal R_i = \emptyset,\ \mathcal A_i\neq\emptyset \\[2pt]"
            r"\theta_i & \text{sinon}"
            r"\end{cases}",
            color=CHARCOAL, font_size=28,
        ).next_to(update, DOWN, buff=0.5).align_to(update, LEFT)

        self.play(FadeIn(img), run_time=0.8)
        self.play(Write(update), run_time=1.0)
        self.play(Write(zonal), run_time=1.8)
        self.wait(2.8)
        self.play(FadeOut(Group(img, update, zonal)), run_time=0.8)


# --- Section 4: Headline metric FSS result ----------------------------

class MetricResult(Scene):
    def construct(self):
        img = fig_image("fig_fss.png", height=6.0)
        img.shift(UP * 0.3)
        self.play(FadeIn(img), run_time=1.0)

        eq2 = MathTex(
            r"\alpha=2:\ \chi_{\max}\sim L^{1.41\pm 0.20}",
            color=BLUE_C, font_size=32,
        )
        eq1 = MathTex(
            r"\alpha=1:\ \chi_{\max}\sim L^{0.97\pm 0.11}",
            color=ORANGE_C, font_size=32,
        )
        eqs = VGroup(eq2, eq1).arrange(RIGHT, buff=1.0).to_edge(DOWN, buff=0.4)
        self.play(Write(eqs), run_time=1.4)
        self.wait(2.8)
        self.play(FadeOut(Group(img, eqs)), run_time=0.8)


# --- Section 5: Bulk fluid (GNF figure) -------------------------------

class BulkScene(Scene):
    def construct(self):
        img = fig_image("fig_gnf.png", height=5.8)
        img.shift(UP * 0.3)
        self.play(FadeIn(img), run_time=1.0)

        eq = MathTex(
            r"\mathrm{Var}(N_\ell) \sim \langle N_\ell\rangle^{2\zeta},"
            r"\qquad \zeta \simeq 0.67\text{--}0.68",
            color=CHARCOAL, font_size=36,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(eq), run_time=1.4)
        self.wait(2.6)
        self.play(FadeOut(Group(img, eq)), run_time=0.8)


# --- Section 6: V-fixed FSS calibration -------------------------------

class CalibrationScene(Scene):
    def construct(self):
        img = fig_image("fig_calibrated.png", height=4.6)
        img.shift(UP * 0.8)
        self.play(FadeIn(img), run_time=1.0)

        vdef = MathTex(
            r"V \equiv 1 - \big|\,\mathbb E[e^{i\xi}]\,\big|",
            color=CHARCOAL, font_size=32,
        ).move_to([0, -1.4, 0])

        res2 = MathTex(
            r"\alpha=2:\ \chi_{\max}\sim L^{1.13\pm 0.13}",
            color=BLUE_C, font_size=30,
        )
        res1 = MathTex(
            r"\alpha=1:\ \chi_{\max}\sim L^{0.23\pm 0.20}",
            color=ORANGE_C, font_size=30,
        )
        res = VGroup(res2, res1).arrange(RIGHT, buff=0.8).to_edge(DOWN, buff=0.4)

        self.play(Write(vdef), run_time=1.0)
        self.play(Write(res), run_time=1.4)
        self.wait(2.6)
        self.play(FadeOut(Group(img, vdef, res)), run_time=0.8)


# --- Section 7: Robustness across model variants ----------------------

class RobustScene(Scene):
    def construct(self):
        img = fig_image("fig_robustness.png", height=5.8)
        img.shift(UP * 0.3)
        self.play(FadeIn(img), run_time=1.0)

        eq = MathTex(
            r"\text{Vicsek standard:}\quad "
            r"L^{1.38\pm 0.11}\ \text{vs}\ L^{0.93\pm 0.20}",
            color=CHARCOAL, font_size=30,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(eq), run_time=1.4)
        self.wait(2.4)
        self.play(FadeOut(Group(img, eq)), run_time=0.8)


# --- Section 8: Topological reversal ----------------------------------

class TopologicalScene(Scene):
    def construct(self):
        img = fig_image("fig_topological.png", height=5.8)
        img.shift(UP * 0.3)
        self.play(FadeIn(img), run_time=1.0)

        line1 = MathTex(
            r"\alpha=1:\ \chi_{\max}\sim L^{1.58\pm 0.66}",
            color=ORANGE_C, font_size=30,
        )
        line2 = MathTex(
            r"\alpha=2:\ \chi_{\max}\sim L^{0.53\pm 0.86}",
            color=BLUE_C, font_size=30,
        )
        eqs = VGroup(line1, line2).arrange(RIGHT, buff=1.0).to_edge(DOWN, buff=0.4)
        self.play(Write(eqs), run_time=1.4)
        self.wait(2.6)
        self.play(FadeOut(Group(img, eqs)), run_time=0.8)


# --- Section 9: Order-parameter PDFs ---------------------------------

class OrderPdfScene(Scene):
    def construct(self):
        img = fig_image("fig_orderpdf.png", height=6.2)
        self.play(FadeIn(img), run_time=1.0)
        eq = MathTex(
            r"P(\varphi)\!:\ \text{bimodal}\ (\alpha=2)\ \to\ "
            r"\text{unimodal}\ (\alpha=1)",
            color=CHARCOAL, font_size=30,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(eq), run_time=1.2)
        self.wait(2.6)
        self.play(FadeOut(Group(img, eq)), run_time=0.8)


# --- Section 10: Adaptive variant -------------------------------------

class AdaptiveScene(Scene):
    def construct(self):
        idea = MathTex(
            r"\alpha_i(t) = \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\,"
            r"\sigma\!\big(s\,(n_i(t)-n_\star)\big)",
            color=CHARCOAL, font_size=30,
        ).to_edge(UP, buff=0.4)

        img = fig_image("fig_adaptive_pilot.png", height=4.8)
        img.shift(DOWN * 0.4)

        result = MathTex(
            r"\text{adaptatif: }L^{0.84\pm 0.31}\quad"
            r"\text{vs Cauchy fixe: }L^{0.55\pm 0.21}",
            color=CHARCOAL, font_size=28,
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(idea), run_time=1.4)
        self.play(FadeIn(img), run_time=1.0)
        self.play(Write(result), run_time=1.4)
        self.wait(2.4)
        self.play(FadeOut(Group(idea, img, result)), run_time=0.8)


# --- Section 11: Synthesis figure ------------------------------------

class SynthesisScene(Scene):
    def construct(self):
        img = fig_image("fig_synthesis.png", height=6.4)
        self.play(FadeIn(img), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(img), run_time=0.8)


# --- Section 12: Closing ---------------------------------------------

class ClosingScene(Scene):
    def construct(self):
        eq = MathTex(
            r"\boxed{\ \alpha<2 \ \Longrightarrow\ "
            r"\chi_{\max}/L^d \to 0\ }",
            color=CHARCOAL, font_size=56,
        )
        ref = styled_text(
            "Code & data: 10.5281/zenodo.20012619",
            font_size=20, color=GRAY,
        ).next_to(eq, DOWN, buff=0.8)

        self.play(Write(eq), run_time=1.6)
        self.play(FadeIn(ref, shift=UP * 0.1), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(VGroup(eq, ref)), run_time=0.8)


# --- Master scene -----------------------------------------------------

class FullExposition(Scene):
    """Concatenate every named slide into one continuous render."""

    def construct(self):
        scenes = [
            TitleScene,
            QuestionScene,
            ModelScene,
            MetricResult,
            BulkScene,
            CalibrationScene,
            RobustScene,
            TopologicalScene,
            OrderPdfScene,
            AdaptiveScene,
            SynthesisScene,
            ClosingScene,
        ]
        for cls in scenes:
            cls.construct(self)
