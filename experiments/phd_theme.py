import plotnine as p9


def phd_theme():
    return p9.theme(
        axis_line=p9.element_line(color="black", size=0.5),
        axis_ticks=p9.element_line(size=0.5),
        axis_title_x=p9.element_text(size=14),
        axis_title_y=p9.element_text(size=14),
        strip_text=p9.element_text(size=14),
        axis_text=p9.element_text(size=11),
        panel_background=p9.element_blank(),
        panel_border=p9.element_blank(),
        panel_grid=p9.element_blank(),
        legend_key=p9.element_blank(),
        legend_background=p9.element_blank(),
        legend_direction="vertical",
        aspect_ratio=0.6,
    )


def phd_theme_bigger():
    return phd_theme() + p9.theme(
        axis_title_x=p9.element_text(size=18),
        axis_title_y=p9.element_text(size=18),
        axis_text=p9.element_text(size=13),
        strip_text=p9.element_text(size=18),
    )
