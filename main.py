import dearpygui.dearpygui as dpg


def print_me(sender):
    print(f"Menu Item: {sender}")


dpg.create_context()


with dpg.window(tag="Primary Window"):
    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
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
    dpg.add_button(label="File Selector")
    dpg.add_text("Directory Path: ")
    dpg.group(horizontal=True)
    dpg.add_text("##filedir", source="directory")
    dpg.add_text("File Path: ")
    dpg.group(horizontal=True)
    dpg.add_text("##file", source="file_directory")
    dpg.add_progress_bar(label="progress", default_value=0.75)


dpg.create_viewport(title='GUNNI', width=600, height=200)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
