import mitsuba as mt
import drjit as dj
mt.set_variant("scalar_rgb")

scene = mt.load_file("/home/jing/PycharmProjects/heliostat_measure/my_scene/my_scene.xml")
image = mt.render(scene, spp=256)

import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2)) # approximate sRGB tonemapping

mt.util.write_bitmap("my_first_render.png", image)
mt.util.write_bitmap("my_first_render.exr", image)