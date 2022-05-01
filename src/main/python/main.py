import dearpygui.dearpygui as dpg
from pathlib import Path
from math import sin
import pandas as pd
import numpy as np
import torch

import nntemplate as mynn


def print_me(sender):
    print(f"Menu Item: {sender}")


def load_model():
    global model
    model = torch.load(dpg.get_value("model_path"), map_location=torch.device('cpu'))


def del_model():
    print(model)



def del_table():
    if dpg.does_item_exist("csv_table"):
        dpg.delete_item("csv_table")


def openfile(sender, app_data):
    # print("Sender: ", sender)
    print("App Data: ", app_data)
    f = open(str(list(app_data['selections'].values())[0]), mode='r')

    content = f.read()
    dpg.set_value("code", content)
    f.close()


def build_csv(sender, app_data):
    csv = pd.read_csv(dpg.get_value("csv_path"))
    csv.index = np.arange(len(csv))
    if dpg.does_item_exist("csv_table"):
        dpg.delete_item("csv_table")
    with dpg.table(parent="csv_view", header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                   tag="csv_table", row_background=True, scrollX=True, scrollY=True):

        # add header columns
        for i in csv.columns:
            dpg.add_table_column(label=f"{i}({csv[i].nunique()})")

        # add rows and cells
        for i in csv.index:
            with dpg.table_row(parent="csv_table"):
                for j in csv.columns:
                    dpg.add_text(csv.at[i, j])


def print_val(sender):
    print(dpg.get_value(sender))


class Browser:
    def __init__(self, dir_sel, tag_parent, exts):
        with dpg.file_dialog(directory_selector=dir_sel, show=False, callback=self.callback, tag=tag_parent,
                             width=720, height=480, default_path=str(Path('..', '..', 'file_samples'))):
            dpg.add_file_extension("", color=(150, 255, 150, 255))
            for ext in exts:
                dpg.add_file_extension(ext, color=exts[ext])

            with dpg.group(horizontal=True):
                dpg.add_button(label="fancy file dialog")
                dpg.add_button(label="file")
                dpg.add_button(label="dialog")
            with dpg.child_window(height=100):
                dpg.add_selectable(label="bookmark 1")
                dpg.add_selectable(label="bookmark 2")
                dpg.add_selectable(label="bookmark 3")

    def callback(self, *args):
        pass


class DatasetBrowser(Browser):
    def __init__(self, tag_child):
        Browser.__init__(self, True, tag_child, {".*": (0, 255, 0, 255)})

    def callback(self, sender, app_data):
        # print(app_data)
        dpg.set_value("dataset_path", list(app_data['selections'].values())[0])
        dpg.set_value("dataset_name", app_data['file_name']) # TODO!!!


class ModelBrowser(Browser):
    def __init__(self, tag_child):
        Browser.__init__(self, False, tag_child, {".pth": (255, 255, 0, 255)})

    def callback(self, sender, app_data):
        dpg.set_value("model_path", list(app_data['selections'].values())[0])
        dpg.set_value("model_name", app_data['file_name'])


class CSVBrowser(Browser):
    def __init__(self, tag_child):
        Browser.__init__(self, False, tag_child, {".csv": (255, 0, 0, 255)})

    def callback(self, sender, app_data):
        print("App Data: ", app_data)
        # f = open(str(list(app_data['selections'].values())[0]), mode='r')
        # content = f.read()
        dpg.set_value("csv_path", list(app_data['selections'].values())[0])
        dpg.set_value("csv_name", app_data['file_name'])
        # f.close()


def gui():
    dpg.create_context()

    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        default_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Light.ttf')), 20)
        second_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Medium.ttf')), 25)

    # THEME DESCRIPTION
    with dpg.theme() as tab_theme:

        with dpg.theme_component(dpg.mvAll):
            # dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 140, 23), category=dpg.mvThemeCat_Core)
            # dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0, category=dpg.mvThemeCat_Core)

    dpg.bind_theme(tab_theme)

    CSVBrowser("csv_browse")
    ModelBrowser("model_browse")
    DatasetBrowser("dataset_browse")

    with dpg.value_registry():
        dpg.add_string_value(tag="dataset_path")
        dpg.add_string_value(default_value="Choose dataset directory", tag="dataset_name")

        dpg.add_string_value(tag="csv_path")
        dpg.add_string_value(default_value="choose csv", tag="csv_name")

        dpg.add_string_value(tag="model_path")
        dpg.add_string_value(default_value="choose model", tag="model_name")

    with dpg.window(tag="Primary Window"):
        dpg.bind_font(default_font)
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_button(label="Browse")
                dpg.add_menu_item(label="Save", callback=print_me)
                dpg.add_menu_item(label="Save As", callback=print_me)

                with dpg.menu(label="Settings"):
                    dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
                    dpg.add_menu_item(label="Setting 2", callback=print_me)

            dpg.add_menu_item(label="Help", callback=print_me)

            with dpg.menu(label="Widget Items"):
                dpg.add_checkbox(label="Pick Me", callback=print_me)
                dpg.add_button(label="Press Me", callback=print_me)
                dpg.add_color_picker(label="Color Me", callback=print_me)
        # dpg.add_text("")

        with dpg.group(horizontal=True):
            with dpg.child_window(pos=[0, 30], label="Settings", width=300, border=True):
                dpg.add_text("NNView", tag="Title")
                dpg.bind_item_font("Title", second_font)
                dpg.add_text("Just text")
                dpg.add_separator()
                dpg.add_progress_bar(label="progress", default_value=0.75)

            with dpg.group(tag="tab group"):
                with dpg.tab_bar():
                    with dpg.tab(label="Model"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("model_browse"))
                            dpg.add_button(label="Load model", callback=load_model)
                            dpg.add_text(source="model_name")
                            dpg.add_button(label="Delete model", callback=del_model)

                    with dpg.tab(label="CSV", tag="csv_view"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("csv_browse"))
                            dpg.add_button(label="Build", callback=build_csv)
                            dpg.add_text(source="csv_name")
                            dpg.add_button(label="Delete", callback=del_table)
                    with dpg.tab(label="Dataset", tag="dataset"):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dataset_browse"))
                            dpg.add_text(source="dataset_name")
                            # dpg.add_button(label="Delete", callback=del_table)
                    with dpg.tab(label="Some plots"):
                        with dpg.child_window(label="Plot", border=False):
                            sindatax = []
                            sindatay = []
                            for i in range(0, 500):
                                sindatax.append(i / 1000)
                                sindatay.append(0.5 + 0.5 * sin(50 * i / 1000))

                            with dpg.plot(label="Line Series", height=400, width=400):
                                # optionally create legend
                                dpg.add_plot_legend()

                                # REQUIRED: create x and y axes
                                dpg.add_plot_axis(dpg.mvXAxis, label="x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")

                                # series belong to a y axis
                                dpg.add_line_series(sindatax, sindatay, label="0.5 + 0.5 * sin(x)", parent="y_axis")


def gui_init():
    dpg.create_viewport(title='NNView', width=1080, height=720)
    dpg.set_viewport_small_icon(str(Path('..', '..', 'assets', 'connect_icon_161112.ico')))
    # dpg.set_viewport_large_icon(str(Path('..', '..', 'assets', 'connect_icon_161112.ico')))

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def main():
    gui()
    gui_init()


if __name__ == '__main__':
    main()
