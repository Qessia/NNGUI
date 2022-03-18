import dearpygui.dearpygui as dpg
from pathlib import Path


def print_me(sender):
    print(f"Menu Item: {sender}")


def callback(sender, app_data):
    print("Sender: ", sender)
    print("App Data: ", app_data)


dpg.create_context()


with dpg.font_registry():
    # first argument ids the path to the .ttf or .otf file
    default_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Light.ttf')), 20)
    second_font = dpg.add_font(str(Path('..', '..', 'assets', 'Roboto-Medium.ttf')), 20)

with dpg.file_dialog(directory_selector=False, show=False, callback=callback, tag="file_dialog_tag",
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
    dpg.add_date_picker()
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
    dpg.add_text("")
    dpg.add_text("Directory Path: ")
    dpg.group(horizontal=True)
    dpg.add_text("##filedir", source="directory")
    dpg.add_text("File Path: ")
    dpg.group(horizontal=True)
    dpg.add_text("##file", source="file_directory")
    dpg.add_progress_bar(label="progress", default_value=0.75)


dpg.create_viewport(title='GUNNI', width=1080, height=720)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
