import dearpygui.dearpygui as dpg
from pathlib import Path
from math import sin


def print_me(sender):
    print(f"Menu Item: {sender}")


def openfile(sender, app_data):
    print("Sender: ", sender)
    print("App Data: ", app_data)
    f = open(str(app_data['selections']), mode='r')
    content = f.read()
    print(content)
    f.close()


def print_val(sender):
    print(dpg.get_value(sender))


def gui():
    dpg.create_context()

    # THEME DESCRIPTION
    with dpg.theme() as tab_theme:

        with dpg.theme_component(dpg.mvAll):
            # dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 140, 23), category=dpg.mvThemeCat_Core)
            # dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0, category=dpg.mvThemeCat_Core)

    dpg.bind_theme(tab_theme)

    with dpg.font_registry():
        # first argument ids the path to the .ttf or .otf file
        default_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Light.ttf')), 20)
        second_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Medium.ttf')), 20)

    with dpg.file_dialog(directory_selector=False, show=False, callback=openfile, tag="file_dialog_tag",
                         width=720, height=480):
        dpg.add_file_extension(".*")
        dpg.add_file_extension("", color=(150, 255, 150, 255))
        dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
        dpg.add_file_extension(".h", color=(255, 0, 255, 255))
        dpg.add_file_extension(".py", color=(0, 255, 0, 255))

        with dpg.group(horizontal=True):
            dpg.add_button(label="fancy file dialog")
            dpg.add_button(label="file")
            dpg.add_button(label="dialog")
        with dpg.child_window(height=100):
            dpg.add_selectable(label="bookmark 1")
            dpg.add_selectable(label="bookmark 2")
            dpg.add_selectable(label="bookmark 3")

    with dpg.window(tag="Primary Window"):
        dpg.bind_font(default_font)
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_tag"))
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

        with dpg.group(pos=[0, 30], horizontal=True):
            with dpg.child_window(label="Setings", width=300, border=True):
                dpg.add_text("<Our powerful app name>")
                dpg.add_text("Directory Path: ")
                dpg.add_text("##filedir", source="directory")
                dpg.add_separator()
                dpg.add_text("File Path: ")
                dpg.add_text("##file", source="file_directory")
                dpg.add_progress_bar(label="progress", default_value=0.75)

            with dpg.tab_bar():
                with dpg.tab(label="Model"):
                    dpg.add_text("Hello")
                with dpg.tab(label="Dataset"):
                    # with dpg.child_window(label="Some text", border=False):
                    dpg.add_text("Work with dataset")
                    dpg.add_button(label="Browse directory")
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
    dpg.create_viewport(title='GUNNI', width=1080, height=720)
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
