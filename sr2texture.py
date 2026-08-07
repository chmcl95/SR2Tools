"""
Turning Sega Rally 2 texture files into Blender images.

texture_classes does the unpacking and color_conversion the colour maths; both
only need struct and numpy, so they run inside Blender as they are. The rest of
the texture tooling goes through PIL, which Blender does not ship, so none of
it is used here - the pixels go straight into a bpy.types.Image instead.
"""
import os

import numpy
import bpy

# Relative when imported as part of the Blender add-on package, absolute when
# this module is run on its own, the way the GUI tool does
try:
    from . import texture_classes
    from . import color_conversion
except ImportError:
    import texture_classes
    import color_conversion


# Signature at the start of a texture file, and the class that reads it
TEXTURE_CLASS_BY_SIGNATURE = {
    b'RTEX': "RTEX",
    b'RHBG': "RHBG",
    b'RTXR': "RTXR",
    b'MTEX': "MTEX",
}

# Tried in order next to the model, first one that exists wins
TEXTURE_EXTENSIONS = (".txr", ".TXR", ".tex", ".TEX")

# The same set for looking at names already on disk, where the case is whatever
# the game shipped - WINDOW.TXR and blur.txr sit in the same folder
LOWERCASE_TEXTURE_EXTENSIONS = {extension.lower() for extension in TEXTURE_EXTENSIONS}

# A car does not ship one texture file per model. Its parts index into a list
# the game builds while loading, and node transforms confirm the order: the
# body meshes of r_corolla ask for 0, the wheels for 1 and the windows for 2.
# Only the clean textures are picked up - the *_dirt* ones are the same images
# with mud on them, swapped in as a race goes on.
CAR_BODY_TEXTURE_NAME = "body"

# ta, de and sn are the tarmac, desert and snow tyres. Tarmac is the plain one
CAR_TYRE_TEXTURE_NAME = "ta_tire"

# The windows come out of the EFFECT folder sitting next to the car folders,
# which every car shares. The game also draws the wind01..wind14 variants kept
# there on a windscreen; WINDOW is the plain one
CAR_EFFECT_FOLDER_NAME = "EFFECT"
CAR_WINDOW_TEXTURE_NAME = "WINDOW"

# "Color Format" of a texture header. Everything else is RGB565
COLOR_FORMAT_ARGB1555 = 2
COLOR_FORMAT_ARGB4444 = 8

# "Palette Usage" of a texture header when the pixels are palette indices
PALETTE_USAGE_INDEXED = 4


def findTextureFile(model_file_path: str):
    """ The texture file sitting next to a model under the same name, if any """
    base_path = os.path.splitext(model_file_path)[0]

    for extension in TEXTURE_EXTENSIONS:
        candidate = base_path + extension
        if os.path.isfile(candidate):
            return candidate

    return None


def openTextureFile(texture_file_path: str):
    """ Read a texture file and return the unpacked SR2Texture """
    with open(texture_file_path, "rb") as texture_file:
        signature = texture_file.read(4)

    class_name = TEXTURE_CLASS_BY_SIGNATURE.get(signature)

    if class_name is None:
        raise ValueError("{} is not a texture file (signature {})".format(
            os.path.basename(texture_file_path), signature))

    texture = getattr(texture_classes, class_name)()
    texture.unpack_from_file(texture_file_path)

    return texture


def paletteForTexture(texture, texture_header):
    """ The palette an indexed texture uses. "Palette Used" counts from 1 """
    palettes = getattr(texture, "palette_list", [])

    if not palettes:
        return None

    if len(palettes) == 1:
        return palettes[0]

    palette_index = texture_header.get("Palette Used", 1) - 1

    if palette_index < 0 or palette_index >= len(palettes):
        print("!!! Palette {} is not in the file, using the first one !!!".format(palette_index + 1))
        palette_index = 0

    return palettes[palette_index]


def decodeTextureToRGBA(texture, texture_index: int):
    """ One texture as an (height, width, 4) array of bytes """
    texture_header = texture.texture_header_list[texture_index]
    pixel_bytes = texture.pixel_bytes_list[texture_index]

    width = texture_header["Image Width"]
    height = texture_header.get("Image Height", width)

    if texture_header.get("Palette Usage") == PALETTE_USAGE_INDEXED:
        palette = paletteForTexture(texture, texture_header)

        if palette is None:
            raise ValueError("texture {} is indexed but the file has no palette".format(texture_index))

        # A palette entry is 4 bytes, blue first, and the fourth is unused -
        # every one of them is 0 across the sample files
        palette_colors = numpy.frombuffer(palette, dtype=numpy.uint8).reshape(-1, 4)
        palette_indices = numpy.frombuffer(pixel_bytes, dtype=numpy.uint8)

        rgba = numpy.empty((len(palette_indices), 4), dtype=numpy.uint8)
        rgba[:, 0] = palette_colors[palette_indices, 2]
        rgba[:, 1] = palette_colors[palette_indices, 1]
        rgba[:, 2] = palette_colors[palette_indices, 0]
        rgba[:, 3] = 255
    elif texture_header["Color Format"] == COLOR_FORMAT_ARGB1555:
        rgba = color_conversion.convertARGB1555toRGBA8888(pixel_bytes)
    elif texture_header["Color Format"] == COLOR_FORMAT_ARGB4444:
        rgba = color_conversion.convertARGB4444toRGBA8888(pixel_bytes)
    else:
        rgba = color_conversion.convertRGB565toRGBA8888(pixel_bytes)

    return numpy.asarray(rgba).reshape(height, width, 4)


def blenderImageFromTexture(texture, texture_index: int, image_name: str):
    """ A packed Blender image holding one texture of the file """
    rgba = decodeTextureToRGBA(texture, texture_index)
    height, width = rgba.shape[0], rgba.shape[1]

    blender_image = bpy.data.images.new(image_name, width=width, height=height, alpha=True)

    # A Blender image starts at its bottom row, the texture at its top one
    flipped = rgba[::-1]
    blender_image.pixels.foreach_set((flipped.astype(numpy.float32) / 255.0).ravel())

    # There is no file to fall back on, so the .blend has to carry the pixels
    blender_image.pack()
    blender_image.update()

    return blender_image


def imagesFromTextureFile(texture_file_path: str):
    """ Every texture of one file, as Blender images, in file order """
    try:
        texture = openTextureFile(texture_file_path)
    except Exception as exception:
        print("!!! Could not read {}: {} !!!".format(os.path.basename(texture_file_path), exception))
        return []

    file_name = os.path.splitext(os.path.basename(texture_file_path))[0]
    images = []

    for texture_index in range(len(texture.texture_header_list)):
        image_name = "{0}_{1:02}".format(file_name, texture_index)

        try:
            images.append(blenderImageFromTexture(texture, texture_index, image_name))
        except Exception as exception:
            print("!!! Could not decode texture {} of {}: {} !!!".format(
                texture_index, os.path.basename(texture_file_path), exception))
            images.append(None)

    print("Loaded {} texture(s) from {}".format(len(images), os.path.basename(texture_file_path)))

    return images


def findCarTextureFile(folder: str, file_name: str):
    """
    A car texture of the folder, by the name the game knows it under.

    Matched without regard to case, because the names come out of the game as
    they were typed - EFFECT holds WINDOW.TXR and blur.txr side by side.
    """
    wanted = file_name.lower()

    for entry in sorted(os.listdir(folder)):
        stem, extension = os.path.splitext(entry)

        if extension.lower() in LOWERCASE_TEXTURE_EXTENSIONS and stem.lower() == wanted:
            return os.path.join(folder, entry)

    return None


def findWindowTexture(car_folder: str):
    """ The shared window texture, in the EFFECT folder next to the car folders """
    effect_folder = os.path.join(os.path.dirname(os.path.normpath(car_folder)),
                                 CAR_EFFECT_FOLDER_NAME)

    if not os.path.isdir(effect_folder):
        return None

    return findCarTextureFile(effect_folder, CAR_WINDOW_TEXTURE_NAME)


def firstImageOfTextureFile(texture_file_path):
    """ The first texture of a file as a Blender image, for the single ones """
    if texture_file_path is None:
        return None

    images = imagesFromTextureFile(texture_file_path)

    return images[0] if images else None


def carTextureList(model_file_path: str):
    """ The textures a car part indexes into, or None if this is not a car """
    folder = os.path.dirname(model_file_path) or "."

    if not os.path.isdir(folder) or findCarTextureFile(folder, CAR_BODY_TEXTURE_NAME) is None:
        return None

    return [
        firstImageOfTextureFile(findCarTextureFile(folder, CAR_BODY_TEXTURE_NAME)),
        firstImageOfTextureFile(findCarTextureFile(folder, CAR_TYRE_TEXTURE_NAME)),
        firstImageOfTextureFile(findWindowTexture(folder)),
    ]


def loadTexturesForModel(model_file_path: str):
    """
    The textures a model's nodes index into, by "Texture Index".

    A Track model ships its own file under the same name (tree_a.mdl and
    tree_a.txr) and the index picks a texture inside it. A car has no such
    file; its parts index into a list the game assembles from the shared
    textures of the car folder, see CAR_TEXTURE_FILE_NAMES.
    """
    texture_file_path = findTextureFile(model_file_path)

    if texture_file_path is not None:
        return imagesFromTextureFile(texture_file_path)

    car_images = carTextureList(model_file_path)

    return car_images if car_images is not None else []


def wireImageIntoMaterial(blender_material, blender_image):
    """ Show the texture on a material, with its alpha honoured """
    blender_material.use_nodes = True

    node_tree = blender_material.node_tree
    shader_node = next((node for node in node_tree.nodes if node.type == 'BSDF_PRINCIPLED'), None)

    image_node = node_tree.nodes.new('ShaderNodeTexImage')
    image_node.image = blender_image
    image_node.location = (-400.0, 300.0)

    if shader_node is None:
        return

    node_tree.links.new(image_node.outputs["Color"], shader_node.inputs["Base Color"])
    node_tree.links.new(image_node.outputs["Alpha"], shader_node.inputs["Alpha"])

    # Removed in Blender 4.2, where the render method covers it instead
    if hasattr(blender_material, "blend_method"):
        blender_material.blend_method = 'BLEND'
