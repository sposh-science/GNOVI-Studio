def test_import_package():
    import gnovi_plot  # noqa: F401


def test_import_app():
    import gnovi_plot.app  # noqa: F401


def test_import_main_window():
    import gnovi_plot.gui.main_window  # noqa: F401


def test_import_plot_canvas():
    import gnovi_plot.gui.widgets.plot_canvas  # noqa: F401


def test_import_dataset_panel():
    import gnovi_plot.gui.widgets.dataset_panel  # noqa: F401


def test_import_dataframe_table_model():
    import gnovi_plot.gui.widgets.dataframe_table_model  # noqa: F401


def test_import_data_package():
    import gnovi_plot.data.dataset  # noqa: F401
    import gnovi_plot.data.dataset_manager  # noqa: F401
    import gnovi_plot.data.importers.text_importer  # noqa: F401
    import gnovi_plot.data.numeric  # noqa: F401


def test_import_import_data_dialog():
    import gnovi_plot.gui.dialogs.import_data_dialog  # noqa: F401
