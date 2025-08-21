############################################################################################################################
# ADAPTED FROM SHAP Waterfall Plot Implementation                                                                          #
# Original Source: https://github.com/shap/shap/blob/ea9e71f4a56d67d7b362d6d2eb9c7871169de5e7/shap/plots/_waterfall.py#L16 #
############################################################################################################################

import re, matplotlib, dataclasses
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from typing import Union
from shap import Explanation

format_str = "%+0.03f"

# Type hints, adapted from matplotlib.typing
RGBColorType = Union[tuple[float, float, float], str]
RGBAColorType = Union[
    str,  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float],
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float],
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float],
]
ColorType = Union[RGBColorType, RGBAColorType, np.ndarray]

@dataclasses.dataclass(frozen=True)
class StyleConfig:
    """A complete set of configuration options for matplotlib-based shap plots."""

    primary_color_positive: ColorType
    primary_color_negative: ColorType
    secondary_color_positive: ColorType
    secondary_color_negative: ColorType
    hlines_color: ColorType
    vlines_color: ColorType
    text_color: ColorType
    tick_labels_color: ColorType

    def asdict(self):
        return dataclasses.asdict(self)


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""
    
    if not issubclass(type(s), str):
        s = format_str % s
    
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


def waterfall(shap_values, max_display=10, show=True, highlight_features=None):

    style = StyleConfig(
        primary_color_positive="#F74D4D",
        primary_color_negative="#3A88FC",
        secondary_color_positive="#D98686",
        secondary_color_negative="#787AE3",
        hlines_color="#cccccc",
        vlines_color="#bbbbbb",
        text_color="white",
        tick_labels_color="#999999",
    )

    labels = {
        "MAIN_EFFECT": "SHAP main effect value for\n%s",
        "INTERACTION_VALUE": "SHAP interaction value",
        "INTERACTION_EFFECT": "SHAP interaction value for\n%s and %s",
        "VALUE": "SHAP value (impact on model output)",
        "GLOBAL_VALUE": "mean(|SHAP value|) (average impact on model output magnitude)",
        "VALUE_FOR": "SHAP value for\n%s",
        "PLOT_FOR": "SHAP plot for %s",
        "FEATURE": "Feature %s",
        "FEATURE_VALUE": "Feature value",
        "FEATURE_VALUE_LOW": "Low",
        "FEATURE_VALUE_HIGH": "High",
        "JOINT_VALUE": "Joint SHAP value",
        "MODEL_OUTPUT": "Model output value",
    }   

    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure the input is an Explanation object
    if not isinstance(shap_values, Explanation):
        emsg = "The waterfall plot requires an `Explanation` object as the `shap_values` argument."
        raise TypeError(emsg)

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed! Perhaps try "
            "`shap.plots.waterfall(shap_values[0])` or for multi-output models, "
            "try `shap.plots.waterfall(shap_values[0, 0])`."
        )
        raise ValueError(emsg)

    base_values = float(shap_values.base_values)
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values))])

    assert all(input_feature in feature_names for input_feature in highlight_features), (
        "All highlight_features must be present in feature_names. "
        f"Provided: {highlight_features}, Available: {feature_names}"
    )

    # init variables we use for tracking the plot locations
    # If highlight_features is provided, ensure those features are always included in the plot
    if highlight_features is not None:
        # Convert feature_names to list for easier indexing
        feature_names_list = list(feature_names) 
        
        # Get top features by absolute SHAP value
        abs_order = np.argsort(-np.abs(values))
        top_indices = list(abs_order[:max_display])
        
        # Find indices for highlight_features not already in top_indices
        highlight_indices = [feature_names_list.index(f) for f in highlight_features if f in feature_names_list and feature_names_list.index(f) not in top_indices]
        # Sort highlight_indices by absolute SHAP value (descending)
        highlight_indices_sorted = sorted(highlight_indices, key=lambda idx: -abs(values[idx]))
        
        # Combine top_indices and sorted highlight_indices (preserving order: top first, then highlights)
        order = np.array(top_indices + highlight_indices_sorted)
        num_features = len(order) + 1
    
    else:
        num_features = min(max_display, len(values))
        order = np.argsort(-np.abs(values))

    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1 

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color=style.vlines_color,
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )

        no_binding_suffix_name = feature_names[order[i]].replace("_binding", "")
        if features is None:
            yticklabels[rng[i]] = no_binding_suffix_name
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f") + " = " + no_binding_suffix_name
                )
            else:
                yticklabels[rng[i]] = str(features[order[i]]) + " = " + no_binding_suffix_name

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = f"{len(shap_values) - num_features + 1} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=style.primary_color_positive,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=style.primary_color_negative,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            dist - hl_scaled,
            0,
            head_length=min(dist, hl_scaled),
            color=style.primary_color_positive,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=style.secondary_color_positive,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], format_str),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], format_str),
                horizontalalignment="left",
                verticalalignment="center",
                color=style.primary_color_positive,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -(-dist - hl_scaled),
            0,
            head_length=min(-dist, hl_scaled),
            color=style.primary_color_negative,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=style.secondary_color_negative,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], format_str),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], format_str),
                horizontalalignment="right",
                verticalalignment="center",
                color=style.primary_color_negative,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(ytick_pos, yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color=style.hlines_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1 / num_features, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    # plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"], fontsize=12, ha="left")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks(
        [base_values + values.sum(), base_values + values.sum() + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax3.set_xticklabels(["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color(style.tick_labels_color)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans)
    )

    tick_labels[1].set_color(style.tick_labels_color)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color(style.tick_labels_color)

    # Draw a thin dotted black line between the last displayed feature and the grouped "other features" row
    if max_display < len(values):
        # The y-tick positions are from num_features-1 (top) to 0 (bottom, "other features")
        # The line should be between the last displayed feature and the grouped row, i.e., between y=0 and y=1
        # So, draw at y=0.5 if "other features" is at y=0, or more generally between the first two y-ticks
        plt.axhline(
            y=0.5,
            color="black",
            linestyle=":",
            linewidth=1,
            zorder=0,
        )
    

    finished_highlighting = []
    
    # Highlight y-tick labels for features in highlight_features
    if highlight_features is not None:
        # Find the yticklabels that correspond to highlight_features, including the last label
        for i, label in enumerate(yticklabels):
           
            # The feature name is after the '=' if present, else the whole label
            if "=" in label:
                feature_name = label.split("=")[-1].strip()
                feature_name = feature_name + "_binding" 

                if feature_name in highlight_features:
                    # Get the tick label object (the second set are just feature names, so use the first set)
                    tick_labels = ax.yaxis.get_majorticklabels()
                    
                    if i < len(tick_labels):
                        tick_label = tick_labels[i]
                        tick_label.set_bbox(dict(facecolor='none', edgecolor='orange', boxstyle='round,pad=1', linewidth=5))

                        finished_highlighting.append(feature_name)    

    assert set(finished_highlighting) == set(highlight_features), (
        "Not all highlight_features were found in the plot. "
        f"Provided: {highlight_features}, Highlighted: {finished_highlighting}"
    )

    if show:
        plt.show()
    else:
        return plt.gca()