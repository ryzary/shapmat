import shap


def summary_plot(shap_values, X, max_display=15, plot_type=None):
    shap.summary_plot(shap_values, X, show=False, plot_size=None,
                      max_display=max_display, plot_type=plot_type)


def waterfall_plot(explainer, shap_df, subject_id, max_display=10):
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_df.loc[subject_id],
                                           show=False, features=shap_df.loc[subject_id], max_display=max_display)
