import dearpygui.dearpygui as dpg

dpg.create_context()

width, height, channels, data = dpg.load_image("..\\..\\file_samples\\gosling-1.jpg")

with dpg.texture_registry(show=True):
    dpg.add_static_texture(width, height, data, tag="texture_tag")

with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()