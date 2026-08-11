import struct
import array
import io
import math
import mathutils

import bmesh
import bpy

# Relative when imported as part of the Blender add-on package, absolute when
# this module is run on its own
try:
    from . import sr2texture
except ImportError:
    import sr2texture


# A node transform's "Texture Index" when the mesh is drawn untextured
NO_TEXTURE_INDEX = -1

# Mesh attribute holding the normals exactly as they were read from the MDL
ORIGINAL_NORMAL_ATTRIBUTE = "sr2_normal"

# Mesh attribute holding the UVs exactly as they were read from the MDL, before
# the V flip that import applies
ORIGINAL_UV_ATTRIBUTE = "sr2_uv"

# Collection property holding the MDL-to-Blender axis conversion of an import,
# flattened row by row. Export reads it back to undo the same conversion.
AXIS_CONVERSION_PROPERTY = "SR2MDL axis conversion"

# Property group holding a mesh's MDL material on a Blender material, so it can
# be edited from the Material tab. Registered by the add-on, see __init__.py
MATERIAL_PROPERTY = "sr2_mdl_material"

# Object property holding the 0x20 block a mesh's Model Pointers "unk_0x18"
# points at, for the meshes that have one
EXTRA_BLOCK_PROPERTY = "Model Extra Block"

# Object properties holding a mesh's face data exactly as it was read, and the
# shape of the Blender mesh built from it. Export puts the bytes back when the
# topology still matches, because not every mesh stores its faces the way this
# tool writes them - see Mesh.unpack_faces_from_bytes
ORIGINAL_FACE_PROPERTY = "Original Face Data"
ORIGINAL_TOPOLOGY_PROPERTY = "Original Topology"

# The material fields, in the order they are packed
MATERIAL_COLOR_0_KEYS = ("R0", "G0", "B0", "A0")
MATERIAL_COLOR_1_KEYS = ("R1", "G1", "B1", "A1")
MATERIAL_FLOAT_KEYS = ("unk_0x08", "unk_0x0C", "unk_0x10", "unk_0x14", "unk_0x18", "unk_0x1C")

# How far a normal may drift from the imported one before it counts as edited
# by the user. Blender's custom normal storage is lossy - it normalizes the
# vectors and quantizes them per corner - so an untouched mesh comes back with
# a small error of its own, which this has to stay above. Re-importing every
# MDL of a car folder puts that error at up to 0.032 on Blender 4.0 and 4.2,
# and up to 0.005 on 4.3 and 5.0. 0.05 is roughly 3 degrees, well below any
# deliberate edit.
NORMAL_EDITED_THRESHOLD = 0.05

# How far a UV may drift from the imported one before it counts as edited.
# Import flips V and export flips it back, which is exact on paper but not in
# single precision, and a texture cares about nothing this small anyway.
UV_EDITED_THRESHOLD = 1e-5

# A node's rotation is three unsigned 16-bit values. Import has always read
# them as pi/0x7FFF radians per unit, which puts a full turn at 0xFFFE, so that
# is what export wraps at. The hardware convention may well be 0x10000 per turn
# instead - the difference is 0.005 degrees either way, but it does mean an
# edited rotation is not guaranteed to come back as the exact same integer.
ROTATION_UNIT_IN_RADIANS = math.pi / 0x7FFF
ROTATION_UNITS_PER_TURN = 2 * 0x7FFF

# How far an object may sit from where it was imported before it counts as
# moved. Node transforms are single precision in the file, so anything above
# the rounding error of that is a real edit.
TRANSFORM_EDITED_THRESHOLD = 1e-5


def fill_dict_from_bytes_by_formatting(dictionary_to_fill: dict, source_bytes: bytes, formatting: str) -> dict:
    value_list = struct.unpack(formatting, source_bytes)
    dictionary_to_fill = dict(zip(dictionary_to_fill.keys(), value_list))
    return dictionary_to_fill


def nodeTransformToMatrix(transform) -> mathutils.Matrix:
    """ Local matrix described by a node transform's position, rotation and scale """
    position = mathutils.Vector((transform["Position X"],
                                 transform["Position Y"],
                                 transform["Position Z"]))

    rotation = mathutils.Euler((transform["Rotation X"] * ROTATION_UNIT_IN_RADIANS,
                                transform["Rotation Y"] * ROTATION_UNIT_IN_RADIANS,
                                transform["Rotation Z"] * ROTATION_UNIT_IN_RADIANS), 'XYZ')

    scale = mathutils.Vector((transform["Scale X"],
                              transform["Scale Y"],
                              transform["Scale Z"]))

    return mathutils.Matrix.LocRotScale(position, rotation, scale)


def matrixIsUnchanged(blender_matrix: mathutils.Matrix, node_matrix: mathutils.Matrix) -> bool:
    """ Whether an object still sits exactly where the node transform put it """
    for row_index in range(4):
        for column_index in range(4):
            blender_value = blender_matrix[row_index][column_index]
            node_value = node_matrix[row_index][column_index]

            scale = max(1.0, abs(blender_value), abs(node_value))

            if abs(blender_value - node_value) > TRANSFORM_EDITED_THRESHOLD * scale:
                return False

    return True


def radiansToNodeRotation(angle_in_radians: float) -> int:
    """ An angle as the unsigned 16-bit value a node transform stores """
    return int(round(angle_in_radians / ROTATION_UNIT_IN_RADIANS)) % ROTATION_UNITS_PER_TURN


def node_has_mesh_data(model_file_bytes: bytes, node: "SR2Node") -> bool:
    """
    Whether a node points at a mesh rather than at a Some Data block.

    Two earlier signals turned out to be unreliable. The field at 0x0C being
    NaN does not mean there is no mesh - it is the Texture Index, and NaN is
    what -1 looks like when read as a float. Neither does the Model
    Pointers' "Vertex Offset" being 0x00 or 0xFFFFFF: the Some Data of
    tenkougen.mdl and pl01-pl03.mdl starts with what looks like a colour
    (0x00FF4100, 0x00F0F0C8, ...) and passes that test as a mesh.

    The node relation's "unk_0x08" does hold. Over the 189 nodes of the
    sample files it is 0 for all 168 nodes that have a mesh and 1 for all
    21 that have Some Data instead.
    """
    if node.relation["unk_0x08"] == 1:
        return False

    # Offset 0x00 is the file header, so it can never hold Model Pointers.
    # An offset past the end of the file means the node is not pointing at
    # anything usable either, which happens with mesh-less nodes of a file
    # that was re-exported after the model was made smaller.
    model_pointers_offset = node.transform["Model Pointers Offset"]

    return 0 < model_pointers_offset and model_pointers_offset + 0x20 <= len(model_file_bytes)



SR2MDL_file_header_dict = {
    "File Size": 0,
    "Header Size": 0x20,
    "Relation Offset": 0,
    "Road count": 0,

    "Node Indexes Size In Bytes": 0,
    "Kinda Pointers Count": 0,
    "Unk4": 0,
    "Unk5": 0,
}

SR2MDL_material = {
    # Color 0
    "R0": 0,
    "G0": 0,
    "B0": 0,
    "A0": 0,

    # Color 1
    "R1": 0,
    "G1": 0,
    "B1": 0,
    "A1": 0,

    "unk_0x08": 0.0,
    "unk_0x0C": 0.0,
    "unk_0x10": 0.0,

    "unk_0x14": 0.0,
    "unk_0x18": 0.0,
    "unk_0x1C": 0.0,
}

SR2MDL_vertex = {
    "Position": [0.0, 0.0, 0.0],
    "Normal": [0.0, 0.0, 0.0],
    "UV": [0.0, 0.0]
}

SR2MDL_model_pointers = {
    "Vertex Offset": 0,
    "Face Offset": 0,
    "Material Offset": 0,

    "Vertex Count": 0,
    "Face Count": 0,
    "Material Count": 0,
    "unk_0x18": 0,
    "unk_0x1C": 0,
}

SR2MDL_draw_options = {
    "unk_0x00": 0,
    "unk_0x04": 0,

    "unk_0x08": 0.0,
    "unk_0x0C": 0.0,
    "unk_0x10": 0.0,
    "unk_0x14": 0.0,

    "unk_0x18": 0,
    "unk_0x1C": 0,
}

SR2MDL_node_transform = {
    "Model Pointers Offset": 0,
    "Draw Options Offset": 0,
    "Node Index": 0,
    # Which texture of the model's texture list the mesh uses, -1 for none.
    # Stored where five floats used to be read, which is why it looked like NaN
    # (-1) or a denormal so small it printed as 1.4e-45 (1) before
    "Texture Index": NO_TEXTURE_INDEX,

    "unk_0x10": 0.0,
    "unk_0x14": 0.0,
    "unk_0x18": 0.0,
    "unk_0x1C": 0.0,

    "Position X": 0.0,
    "Position Y": 0.0,
    "Position Z": 0.0,
    "unk_0x2C": 0,

    "Rotation X": 0,
    "unk_0x32": 0,
    "Rotation Y": 0,
    "unk_0x36": 0,
    "Rotation Z": 0,
    "unk_0x3A": 0,
    "unk_0x3C": 0,

    "Scale X": 0.0,
    "Scale Y": 0.0,
    "Scale Z": 0.0,
    "unk_0x4C": 0,

    "unk_0x50": 0,
    "unk_0x54": 0,
    "unk_0x58": 0,
    "unk_0x5C": 0,
}

SR2MDL_node_relation = {
    "Child Offset": 0,
    "Sibling Offset": 0,
    "unk_0x08": 0,
    "unk_0x0C ": 0,

    "unk_0x10 ": 0,
    "unk_0x14 ": 0,
    "unk_0x18 ": 0,
    "unk_0x1C ": 0,
}

SR2MDL_Road = {
    "Index": 0,
    "Node Offset": 0,
    "Node Offset LOD": 0,
    "Unk_3": 0,

    "Unk_4": 0,
    "Unk_5": 0,
    "Unk_6": 0,
    "Unk_7": 0,

    "Unk_8": 0,
    "Unk_9": 0,
    "Unk_10": 0,
    "Unk_11": 0,

    "Unk_12": 0,
    "Unk_13": 0,
    "Unk_14": 0,
    "Unk_15": 0,

    "Unk_16": 0,
    "Unk_17": 0,
    "Unk_18": 0,
    "Unk_19": 0,

    "Embed Texture Offset": 0,
    "Unk_21": 0,
    "Unk_22": 0,
    "Unk_23": 0,
}


SR2MDL_kinda_pointers = {
    "Unk_0": 0,
    "Unk_1": 0,
    "Unk_2": 0,
    "Unk_3": 0,

    "Unk_4": 0,
    "Unk_5": 0,
    "Unk_6": 0,
    "Unk_7": 0,
}


class Vertex:
    format = '<8f'
    
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.normal = [0.0, 0.0, 0.0]
        self.uv = [0.0, 0.0]

    def unpack_from_float_bytes(self, eight_floats_in_bytes):
        float_list = struct.unpack_from(self.format, eight_floats_in_bytes, 0)

        self.position = [float_list[0], float_list[1], float_list[2]]
        self.normal = [float_list[3], float_list[4], float_list[5]]
        self.uv = [float_list[6], float_list[7]]


class Mesh:
    material_format = "<8B" + "6f"
    format_face = '<{0}h'
    format_model_pointers = '<8I'
    draw_options_format = '<2I' + '4f' + '2I'

    # The block a mesh's Model Pointers "unk_0x18" points at, read as raw
    # 32-bit words so any bit pattern comes back out the way it went in
    extra_block_format = '<8i'
    extra_block_size = 0x20

    def __init__(self):
        self.offset = 0
        self.total_size = 0

        self.material = {}
        self.vertexes = []
        self.faces = []
        self.model_pointers = {}
        self.draw_options = {}

        # None when the mesh has no such block, which is all but 73 of the
        # 3810 meshes of the sample files - see unpack_extra_block_from_bytes
        self.extra_block = None

        # The face region as it was read, padding and all. Written back
        # untouched when the user did not change the topology
        self.original_face_bytes = None

        # Used for packing
        self.sizes = {
            "Material": 0x20,
            "Vertex": 0,
            "Face": 0,
            "ExtraBlock": 0,
            "ModelPointers": 0x20,
            "DrawOptions": 0x20
        }

    def update_sizes(self):
        vertex_size = len(self.vertexes) * 4 * 8

        # Add padding to face size, so it would be aligned to 32-bytes
        if self.original_face_bytes is not None:
            face_size = len(self.original_face_bytes)
        else:
            face_size = len(self.faces) * 2
            if face_size % 32 != 0:
                face_size = (((face_size//32) + 1) * 32)

        extra_block_size = self.extra_block_size if self.extra_block is not None else 0

        self.sizes["Vertex"] = vertex_size
        self.sizes["Face"] = face_size
        self.sizes["ExtraBlock"] = extra_block_size

        self.total_size = 0x20 + vertex_size + face_size + extra_block_size + 0x20 + 0x20

    """ Unpacking """

    def unpack_material_from_bytes(self, full_model_file_bytes, material_offset):
        if self.model_pointers["Material Count"] > 1:
            print("!!! MULTIPLE MATERIALS PRESENT !!!... or the data was read incorrectly")

        material_size_in_bytes = self.model_pointers["Material Count"] * 0x20
        material_bytes = full_model_file_bytes[material_offset:
                                               material_offset + material_size_in_bytes]

        self.material = fill_dict_from_bytes_by_formatting(SR2MDL_material,
                                                           material_bytes,
                                                           self.material_format)

    def unpack_vertexes_from_bytes(self, full_model_file_bytes, vertex_offset, vertex_count):
        # One vertex contains 8 floats (3 position, 3 normal, 2 uv)
        vertex_size_in_bytes = 32  # 4*3 + 4*3 + 4*2
        vertex_array_size_in_bytes = vertex_count * vertex_size_in_bytes

        if vertex_offset > 0 and vertex_offset != 0xffffff:
            vertex_bytes = full_model_file_bytes[vertex_offset:
                                                 vertex_offset + vertex_array_size_in_bytes]

            for vertex_index in range(0, vertex_count):
                vertex = Vertex()
                current_vertex_offset = vertex_index * vertex_size_in_bytes
                vertex.unpack_from_float_bytes(vertex_bytes[current_vertex_offset:
                                                            current_vertex_offset + vertex_size_in_bytes])
                self.vertexes.append(vertex)

    def unpack_faces_from_bytes(self, full_model_file_bytes, face_offset, face_count):
        # A face (triangle) is defined by 3 vertices. Vertex index is only 2 bytes long
        one_face_size_in_bytes = 6  # 2*3
        face_array_size_in_bytes = face_count * one_face_size_in_bytes

        face_bytes = full_model_file_bytes[face_offset:
                                           face_offset + face_array_size_in_bytes]

        fmt_faces = self.format_face.format(face_count)
        self.faces = struct.unpack_from(fmt_faces, face_bytes, 0)

        # Keep the region as it stands when writing the indices back out would
        # not reproduce it.
        #
        # "Face Count" is an index count for the usual mesh, always a multiple
        # of three, and those survive the trip through Blender unchanged. 98
        # meshes of the sample files hold 1 instead, every one of them a
        # four-vertex billboard, and their region is four 32-bit indices - 3,
        # 2, 1, 0 or 1, 0, 3, 2 - sometimes followed by a 1.0 or -1.0. What
        # picks one encoding over the other is not worked out yet.
        #
        # Import turns such a mesh into a Blender quad, and writing that back
        # as six 16-bit indices with a Face Count of 6 froze the game on the
        # car light models. Putting the original bytes back avoids guessing.
        padded_size = face_count * 2
        if padded_size % 32 != 0:
            padded_size = ((padded_size // 32) + 1) * 32

        if face_offset <= 0 or face_offset + padded_size > len(full_model_file_bytes):
            return

        region = full_model_file_bytes[face_offset:face_offset + padded_size]

        if region != self.pack_faces():
            self.original_face_bytes = region

    def unpack_model_pointers_from_bytes(self, full_model_file_bytes, model_pointer_offset):
        model_pointers_size_in_bytes = 0x20
        model_pointers_bytes = full_model_file_bytes[model_pointer_offset:
                                                     model_pointer_offset + model_pointers_size_in_bytes]

        self.model_pointers = fill_dict_from_bytes_by_formatting(SR2MDL_model_pointers,
                                                                 model_pointers_bytes,
                                                                 self.format_model_pointers)

    def unpack_draw_options_from_bytes(self, full_model_file_bytes, draw_options_offset):
        draw_option_size_in_bytes = 0x20

        draw_option_bytes = full_model_file_bytes[draw_options_offset:
                                                  draw_options_offset + draw_option_size_in_bytes]

        self.draw_options = fill_dict_from_bytes_by_formatting(SR2MDL_draw_options,
                                                               draw_option_bytes,
                                                               self.draw_options_format)

    def unpack_extra_block_from_bytes(self, full_model_file_bytes, model_pointer_offset):
        """
        Read the 0x20 block sitting between the faces and the Model Pointers.

        "unk_0x18" holds its offset when there is one. Telling that apart from
        the small number the field otherwise carries is exact: over the 3810
        meshes of the sample files it either equals Model Pointers - 0x20 (73
        meshes, all of them the four-vertex light billboards of a car) or is
        far too small to be an offset at all (the other 3737). Nothing lands in
        between, and in all 73 the block starts the moment the face data ends.

        Every one of the 73 holds the same eight floats, 0.0 and 1.0 four times
        over - a pair per vertex of the quad. What they mean is still open, so
        the bytes are kept as they are and written back untouched. Dropping
        them used to shift everything behind the mesh and change the offsets in
        the node array of 29 sample files.
        """
        extra_block_offset = self.model_pointers["unk_0x18"]

        if extra_block_offset != model_pointer_offset - self.extra_block_size:
            return

        if extra_block_offset < 0 or extra_block_offset + self.extra_block_size > len(full_model_file_bytes):
            return

        self.extra_block = list(struct.unpack_from(self.extra_block_format,
                                                   full_model_file_bytes,
                                                   extra_block_offset))

    def unpack_from_bytes(self, full_model_file_bytes, model_pointer_offset):
        """
        Material
        Vertex[]
        Face[]
        ExtraBlock, when the mesh has one
        ModelPointers
        DrawOptions
        """
        # Unpack Model Pointers first because it has offsets to other things
        self.unpack_model_pointers_from_bytes(full_model_file_bytes, model_pointer_offset)

        self.unpack_material_from_bytes(full_model_file_bytes, self.model_pointers["Material Offset"])

        self.unpack_vertexes_from_bytes(full_model_file_bytes,
                                        self.model_pointers["Vertex Offset"],
                                        self.model_pointers["Vertex Count"])

        self.unpack_faces_from_bytes(full_model_file_bytes,
                                     self.model_pointers["Face Offset"],
                                     self.model_pointers["Face Count"])

        self.unpack_extra_block_from_bytes(full_model_file_bytes, model_pointer_offset)

        # Unpack Draw Options
        model_pointers_size_in_bytes = 0x20
        self.unpack_draw_options_from_bytes(full_model_file_bytes,
                                            model_pointer_offset + model_pointers_size_in_bytes)

    """ Packing """

    def pack_material(self) -> bytes:
        return struct.pack(self.material_format, *self.material.values())

    def pack_vertexes(self) -> bytes:
        vertex_bytes = b''

        for vertex in self.vertexes:
            position_bytes = struct.pack("<3f", *vertex.position)
            normal_bytes = struct.pack("<3f", *vertex.normal)
            uv_bytes = struct.pack("<2f", *vertex.uv)

            vertex_bytes += position_bytes + normal_bytes + uv_bytes

        return vertex_bytes

    def pack_faces(self) -> bytes:
        # An untouched mesh goes back exactly as it came in, whatever encoding
        # its faces were in - see unpack_faces_from_bytes
        if self.original_face_bytes is not None:
            return bytes(self.original_face_bytes)

        # Faces need to be packed with 32-byte alignment
        face_format_with_amount = self.format_face.format(len(self.faces))
        face_bytes = struct.pack(face_format_with_amount, *self.faces)

        face_bytes_size_in_bytes = len(face_bytes)

        if face_bytes_size_in_bytes % 32 != 0:
            total_size_needed = (((face_bytes_size_in_bytes//32) + 1) * 32)
            padding_needed = total_size_needed - face_bytes_size_in_bytes

            face_bytes += struct.pack("{}x".format(padding_needed))

        return face_bytes

    def pack_model_pointers(self):
        return struct.pack(self.format_model_pointers, *self.model_pointers.values())

    def pack_draw_options(self):
        return struct.pack(self.draw_options_format, *self.draw_options.values())

    def pack_extra_block(self) -> bytes:
        if self.extra_block is None:
            return b''

        return struct.pack(self.extra_block_format, *self.extra_block)

    def pack_and_return(self):
        """
            Model
        Material
        Vertex[]
        Face[]
        ExtraBlock, when the mesh has one
        ModelPointer
        DrawOption
        """
        self.update_sizes()

        material_bytes = self.pack_material()
        vertex_bytes = self.pack_vertexes()
        faces_bytes = self.pack_faces()
        extra_block_bytes = self.pack_extra_block()
        model_pointers_bytes = self.pack_model_pointers()
        draw_options_bytes = self.pack_draw_options()

        return (material_bytes + vertex_bytes + faces_bytes + extra_block_bytes
                + model_pointers_bytes + draw_options_bytes)


some_data = {
    "0": 0.0,
    "1": 0.0,
    "2": 0.0,
    "3": 0.0,

    "4": 0.0,
    "5": 0.0,
    "6": 0.0,
    "7": 0.0,
}


class SomeData:
    format = '<8f'

    def __init__(self):
        self.data = dict.copy(some_data)

    def unpack_from_bytes(self, data_bytes, offset):

        some_data_bytes = data_bytes[offset:
                                     offset + 0x20]

        # Same as in node_has_mesh_data - the offset can point outside the file
        if len(some_data_bytes) < 0x20:
            print("!!! Some Data at {0:#X} is outside the file, skipping it !!!".format(offset))
            return

        self.data = fill_dict_from_bytes_by_formatting(self.data,
                                                       some_data_bytes,
                                                       self.format)

    def pack_and_return(self):
        return struct.pack(self.format, *self.data.values())


SR2Node_extra = {
        "Offset": 0,
        "Relation Offset": 0,
        "Index": 0,
        "Child Index": -1,
        "Sibling Index": -1
}


class SR2Node:
    # The '1i' is the Texture Index, which used to be read as the first of five
    # floats. Same bytes, same size - only the interpretation changes
    format_transform = '<3I' + '1i' + '4f' + '3f' + '1I' + '6H' + '1I' + '3f' + '5I'
    format_relation = '<8I'

    format = '<4I' + '4f' + '3f' + '5I' + '3f' + "5I" + '8I'
    #  4I  4f   4f    4f     3f     20x    8I'
    # 0-3, 4-7, 8-11, 12-15, 16-18, 19-29, 30-38

    def __init__(self):
        self.node_transform_size = 0x60
        self.node_relation_size = 0x20

        self.transform = {}
        self.relation = {}
        self.some_data = {}

        # Mesh belonging to this node, if any (set during unpack; not serialized)
        self.mesh = None

        self.mesh_offset = 0x00
        self.draw_ops_offset = 0x00

        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.scale = [0, 0, 0]

        # Put these values into a dictionary to be able to store it in the blender file
        self.extra = dict.copy(SR2Node_extra)

        """
        self.offset = 0x00
        self.relation_offset = 0x00

        # Index refers to the position in SR2MDL.nodes
        self.index = 0
        self.parent_index = -1
        self.child_index = -1
        """

    def unpack_from_bytes(self, node_bytes):
        self.transform = fill_dict_from_bytes_by_formatting(SR2MDL_node_transform,
                                                            node_bytes[:self.node_transform_size],
                                                            self.format_transform)

        self.relation = fill_dict_from_bytes_by_formatting(SR2MDL_node_relation,
                                                           node_bytes[self.node_transform_size:],
                                                           self.format_relation)

        self.mesh_offset = self.transform["Model Pointers Offset"]
        self.draw_ops_offset = self.transform["Draw Options Offset"]

        self.position = [self.transform["Position X"],
                         self.transform["Position Y"],
                         self.transform["Position Z"]]
        self.rotation = [self.transform["Rotation X"],
                         self.transform["Rotation Y"],
                         self.transform["Rotation Z"]]
        self.scale = [self.transform["Scale X"],
                      self.transform["Scale Y"],
                      self.transform["Scale Z"]]

    def pack_and_return(self):
        node_transform_bytes = b''

        if self.transform != {}:
            node_transform_bytes = struct.pack(self.format_transform, *self.transform.values())

        node_relation_bytes = struct.pack(self.format_relation, *self.relation.values())

        return node_transform_bytes + node_relation_bytes


class SR2RoadSegment:
    format_road = "<24I"

    def __init__(self):
        self.road = dict.copy(SR2MDL_Road)
        self.road_size = 0x60

    # Included for speed
    def unpack_from_road_bytes(self, road_bytes):
        self.road = fill_dict_from_bytes_by_formatting(self.road,
                                                       road_bytes,
                                                       self.format_road)

    def unpack_from_bytes(self, file_bytes, offset):
        road_bytes = file_bytes[offset, offset + self.road_size]

        self.road = fill_dict_from_bytes_by_formatting(self.road,
                                                       road_bytes,
                                                       self.format_road)


class SR2MDL:
    node_indexes_format = "<{}I"
    kinda_pointers_format = "<8I"

    def __init__(self):
        self.file_header = dict.copy(SR2MDL_file_header_dict)
        self.file_header_formatting = '<8I'
        self.file_header_size = struct.calcsize(self.file_header_formatting)

        self.meshes = []
        self.nodes = []

        # Present in level files
        self.roads = []
        self.node_indexes = []
        self.kinda_pointers = []
        self.embedded_textures = []

        self.embedded_textures_bytes = []

    def fill_file_header_from_bytes(self, model_file_bytes):
        file_header_bytes = model_file_bytes[:self.file_header_size]

        self.file_header = fill_dict_from_bytes_by_formatting(self.file_header,
                                                              file_header_bytes,
                                                              self.file_header_formatting)

    def unpack_road_segments_from_bytes(self, model_file_bytes, road_offset, road_count):
        road_size_in_bytes = 0x60

        # All Road bytes, for speed
        road_bytes_chunk = model_file_bytes[road_offset:
                                            road_offset + road_size_in_bytes * road_count]

        for road_offset_in_chunk in range(0, len(road_bytes_chunk), road_size_in_bytes):
            new_road = SR2RoadSegment()

            road_bytes = road_bytes_chunk[road_offset_in_chunk:
                                          road_offset_in_chunk + road_size_in_bytes]

            new_road.unpack_from_road_bytes(road_bytes)

            self.roads.append(new_road)

    def unpack_node_indexes_from_bytes(self, model_file_bytes, offset, node_indexes_size):
        single_index_size = 4
        node_indexes_count = node_indexes_size // single_index_size

        # "Node Indexes Size In Bytes" is not always a multiple of 4 (seen in
        # RIVIERA.DAT: 0x9E = 158 bytes), so only slice the bytes that make up
        # whole indexes here; unpack_level_model_from_bytes still advances to
        # the next section by the raw header value, since that's the real
        # physical stride and any trailing bytes are just unparsed padding.
        aligned_size = node_indexes_count * single_index_size

        node_indexes_bytes = model_file_bytes[offset:
                                              offset + aligned_size]

        node_indexes_formated = self.node_indexes_format.format(node_indexes_count)

        self.node_indexes = struct.unpack(node_indexes_formated, node_indexes_bytes)

    def unpack_kinda_pointers_from_bytes(self, model_file_bytes, offset, kinda_pointers_count):
        """
        Seem to come in groups

        1 value - Pointer to vertex array of a Mesh
        2 value - Pointer to start of face array of the same Mesh

        4 value - A node index? Just index?

        Next in the same group
        1 value - same pointer as before
        2 value - same pointer as before, but slightly larger. Doesn't seem to point to something correctly. Doesn't make sense
        """

        single_kinda_pointers_size = struct.calcsize(self.kinda_pointers_format)

        kinda_pointers_offset = offset
        for i in range(kinda_pointers_count):
            kinda_pointers_bytes = model_file_bytes[kinda_pointers_offset:
                                                    kinda_pointers_offset + single_kinda_pointers_size]

            unpacked_kinda_pointers = struct.unpack(self.kinda_pointers_format, kinda_pointers_bytes)
            self.kinda_pointers.append(unpacked_kinda_pointers)

            kinda_pointers_offset += single_kinda_pointers_size

    def unpack_embedded_textures_from_bytes(self, model_file_bytes, starting_offset):
        # Split by their sizes, actual texture handling should be carried by texture classes

        # Width, Height, pixel bytes (RGB565)
        current_texture_offset = starting_offset
        end_of_file_offset = len(model_file_bytes)

        if current_texture_offset >= end_of_file_offset:
            return

        header_size_in_bytes = 8
        header_formatting = "<2I"

        while True:
            header_bytes = model_file_bytes[current_texture_offset:
                                            current_texture_offset + header_size_in_bytes]

            temp_header = struct.unpack(header_formatting, header_bytes)

            width = temp_header[0]
            height = temp_header[1]

            if (width == 0) or (height == 0):
                break

            pixel_data_size_in_bytes = width * height * 2  # 16 // 8

            texture_bytes = model_file_bytes[current_texture_offset:
                                             current_texture_offset + pixel_data_size_in_bytes + header_size_in_bytes]
            self.embedded_textures_bytes.append(texture_bytes)

            current_texture_offset += header_size_in_bytes + pixel_data_size_in_bytes
            if current_texture_offset >= end_of_file_offset:
                break

    def unpack_level_model_from_bytes(self, model_file_bytes):
        self.fill_file_header_from_bytes(model_file_bytes[:self.file_header_size])

        node_transform_size = 0x60
        node_relation_size = 0x20
        node_size = node_relation_size + node_transform_size

        # Read Road segments
        road_count = self.file_header["Road count"]
        road_size_in_bytes = 0x60
        road_offset = self.file_header["Relation Offset"] + 0x20

        self.unpack_road_segments_from_bytes(model_file_bytes, road_offset, road_count)

        # A road does not point at a single node but at the head of a chain -
        # its siblings are the other pieces of scenery standing on that stretch
        # of track. Following only the head left most of a level behind: 120 of
        # RIVIERA.DAT's 628 nodes, 437 of DES_SS1.DAT's 902. Roads share nodes
        # as well, so the same offset must not be read twice.
        unpacked_node_offsets = set()

        for road in self.roads:
            for node_offset in (road.road["Node Offset"], road.road["Node Offset LOD"]):
                self.unpack_node_chain_by_offset(model_file_bytes, node_offset, unpacked_node_offsets)

        self.find_node_index_relations_by_node_relation_offsets()

        node_indexes_offset = road_offset + road_size_in_bytes * road_count
        self.unpack_node_indexes_from_bytes(model_file_bytes, node_indexes_offset, self.file_header["Node Indexes Size In Bytes"])

        kinda_pointer_offset = node_indexes_offset + self.file_header["Node Indexes Size In Bytes"]
        self.unpack_kinda_pointers_from_bytes(model_file_bytes, kinda_pointer_offset, self.file_header["Kinda Pointers Count"])

        embedded_textures_offset = kinda_pointer_offset + self.file_header["Kinda Pointers Count"] * 32
        self.unpack_embedded_textures_from_bytes(model_file_bytes, embedded_textures_offset)

    def unpack_node_chain_by_offset(self, model_file_bytes, node_offset, unpacked_node_offsets):
        """
        Unpack a node and everything hanging off it.

        A node's relation holds the offset of its first child at 0x00 and of
        its next sibling at 0x04 - see docs/mdl_node_pointer_memo.md.
        """
        pending = [node_offset]

        while pending:
            offset = pending.pop()

            if offset <= 0 or offset + 0x80 > len(model_file_bytes) or offset in unpacked_node_offsets:
                continue

            unpacked_node_offsets.add(offset)

            node = self.unpack_node_by_offset(model_file_bytes, offset)

            pending.append(node.relation["Sibling Offset"])
            pending.append(node.relation["Child Offset"])

    def unpack_node_by_offset(self, model_file_bytes, node_offset):
        new_node = SR2Node()
        new_node.unpack_from_bytes(model_file_bytes[node_offset:node_offset + 0x80])

        new_node.extra["Offset"] = node_offset
        new_node.extra["Relation Offset"] = node_offset + new_node.node_transform_size

        self.nodes.append(new_node)

        # Same as for a car model - a node either describes a mesh or points at
        # a Some Data block, and reading the one as the other gives nonsense
        if node_has_mesh_data(model_file_bytes, new_node):
            new_mesh = Mesh()
            new_mesh.unpack_from_bytes(model_file_bytes, new_node.transform["Model Pointers Offset"])

            new_node.mesh = new_mesh
            self.meshes.append(new_mesh)
        else:
            some_data = SomeData()
            some_data.unpack_from_bytes(model_file_bytes, new_node.transform["Model Pointers Offset"])
            new_node.some_data = some_data.data

        return new_node

    def unpack_from_bytes(self, model_file_bytes):
        self.fill_file_header_from_bytes(model_file_bytes[:self.file_header_size])

        print('File size: {0:#X}'.format(self.file_header["File Size"]))
        print('Node offset: {0:#X}'.format(self.file_header["Relation Offset"]))

        # Check if it's a level file
        if self.file_header["File Size"] > self.file_header["Relation Offset"] + 0x20:
            self.unpack_level_model_from_bytes(model_file_bytes)
            return

        # Unpack node and related mesh
        # 1. Calculate the total size of nodes + mesh
        # 2. Go through each node + mesh and subtract their sizes from the total size
        # 3. Once the total size hits zero, the reading is complete

        node_transform_size = 0x60
        node_relation_size = 0x20
        node_size = node_relation_size + node_transform_size

        total_mesh_node_size = self.file_header["Relation Offset"] + node_relation_size - self.file_header_size

        # Unpack nodes and related mesh, if exist
        # A node's Texture Index is -1 when its mesh is drawn untextured

        first_node_relation_offset = self.file_header["Relation Offset"]

        # MDL file header has the first node relation offset
        # Nodes after that are placed before the first one

        bytes_left = total_mesh_node_size
        current_node_relation_offset = first_node_relation_offset

        while bytes_left > 0:
            print("\n")
            print("Total bytes left : {}".format(bytes_left))

            mesh_present = True

            # Unpack node
            node = SR2Node()

            node_bytes = model_file_bytes[current_node_relation_offset - node_transform_size:
                                          current_node_relation_offset - node_transform_size + node_size]
            node.unpack_from_bytes(node_bytes)

            print("Texture Index {0}".format(node.transform["Texture Index"]))

            if not node_has_mesh_data(model_file_bytes, node):
                print("Node without Mesh detected")
                mesh_present = False

            node.index = len(self.nodes)  # len(self.nodes) will increase as Nodes are added

            node.extra["Offset"] = current_node_relation_offset - node_transform_size
            node.extra["Relation Offset"] = current_node_relation_offset

            self.nodes.append(node)

            # The topmost node in the hierarchy has neither a parent nor a child
            # relation. Some files contain bytes between the last real node's data
            # and the node array that aren't accounted for by the size bookkeeping
            # below, so bytes_left is not guaranteed to reach exactly 0 once this
            # node is processed. Stop here instead of reading past the real data.
            is_last_node = (node.relation["Child Offset"] == 0
                             and node.relation["Sibling Offset"] == 0)

            # Go back through the file
            current_node_relation_offset -= node_size
            bytes_left -= node_size

            # Unpack mesh
            if mesh_present:
                mesh_offset = node.mesh_offset

                mesh = Mesh()
                mesh.unpack_from_bytes(model_file_bytes, mesh_offset)
                mesh.update_sizes()

                bytes_left -= mesh.total_size

                node.mesh = mesh
                self.meshes.append(mesh)

                print("Mesh at {0:#X}".format(current_node_relation_offset))
                print("Mesh size: {0:#X}".format(mesh.total_size))
            else:
                some_data_size = 0x20
                tmp_some_data = SomeData()

                tmp_some_data.unpack_from_bytes(model_file_bytes, node.transform["Model Pointers Offset"])

                # Attach to node, instead of array like mesh
                node.some_data = tmp_some_data.data

                bytes_left -= some_data_size

            print("Total bytes left after substraction: {}".format(bytes_left))

            if is_last_node:
                break

        self.find_node_index_relations_by_node_relation_offsets()

        for node in self.nodes:
            print("\n")
            print("Node Index", node.transform["Node Index"])
            print("Child Index", node.extra["Child Index"])
            print("Sibling Index", node.extra["Sibling Index"])

    def find_node_index_relations_by_node_relation_offsets(self):
        # Find Node child and sibling by index for easy offset calculation.
        # A level has over a thousand nodes, so this looks the offsets up in a
        # dict rather than comparing every node against every other one
        print("\nFinding Relationships between Nodes")

        index_by_offset = {node.extra["Offset"]: index for index, node in enumerate(self.nodes)}

        for node in self.nodes:
            node.extra["Child Index"] = index_by_offset.get(node.relation["Child Offset"], -1)
            node.extra["Sibling Index"] = index_by_offset.get(node.relation["Sibling Offset"], -1)

    def unpack_from_file(self, file_path):
        with open(file_path, "r+b") as file:
            file_bytes = file.read()
            self.unpack_from_bytes(file_bytes)

    def calculate_total_node_size(self):
        return len(self.nodes) * 0x80

    def update_sizes(self):
        """
        Header
        Mesh[]
        Node[]
        """

        for mesh in self.meshes:
            mesh.update_sizes()

        total_mesh_size = 0
        for mesh in self.meshes:
            total_mesh_size += mesh.total_size

        total_node_size = self.calculate_total_node_size()

        total_some_data_size = 0

        for node in self.nodes:
            if node.some_data != {}:
                total_some_data_size += 0x20

        total_file_size = self.file_header_size + total_mesh_size + total_some_data_size + total_node_size

        self.file_header["File Size"] = total_file_size

    def update_relation_offset_in_header(self):
        self.update_sizes()

        first_node_relation_offset = self.file_header_size

        for mesh in self.meshes:
            first_node_relation_offset += mesh.total_size

        for node in self.nodes:
            if node.some_data != {}:
                first_node_relation_offset += 0x20

        # Go the end of all nodes and then go back a little
        first_node_relation_offset += self.calculate_total_node_size()
        first_node_relation_offset -= 0x20

        self.file_header["Relation Offset"] = first_node_relation_offset

    def update_model_pointers_offsets(self):
        current_mesh_offset = self.file_header_size
        for mesh in self.meshes:
            mesh.offset = current_mesh_offset

            mesh.model_pointers["Vertex Offset"] = current_mesh_offset + mesh.sizes["Material"]
            mesh.model_pointers["Face Offset"] = current_mesh_offset + mesh.sizes["Material"] + mesh.sizes["Vertex"]
            mesh.model_pointers["Material Offset"] = current_mesh_offset

            # The extra block follows the faces, so its offset moves with them.
            # A mesh without one keeps whatever small number the field held
            if mesh.extra_block is not None:
                mesh.model_pointers["unk_0x18"] = (current_mesh_offset + mesh.sizes["Material"]
                                                   + mesh.sizes["Vertex"] + mesh.sizes["Face"])

            current_mesh_offset += mesh.total_size

    def update_node_offset_to_model_pointers(self):
        # Nodes without a mesh can appear anywhere in the list, not just at the
        # end, so self.meshes can't be indexed positionally by node_index -
        # use the node<->mesh association instead.
        # Some Data blocks follow all the meshes, in node order - see pack_and_return
        some_data_offset = self.file_header_size
        for mesh in self.meshes:
            some_data_offset += mesh.total_size

        for node in self.nodes:
            if node.mesh is not None:
                mesh = node.mesh

                node.transform["Model Pointers Offset"] = (mesh.offset + mesh.sizes["Material"]
                                                           + mesh.sizes["Vertex"] + mesh.sizes["Face"]
                                                           + mesh.sizes["ExtraBlock"])
                node.transform["Draw Options Offset"] = node.transform["Model Pointers Offset"] + 0x20
            elif node.some_data != {}:
                # A mesh-less node points at its Some Data instead. Leaving the
                # offset from the file it was imported from makes it dangle as
                # soon as anything before it changes size.
                node.transform["Model Pointers Offset"] = some_data_offset
                some_data_offset += 0x20

    def update_node_offsets(self):
        self.update_node_offset_to_model_pointers()
        pass

    def update_offsets(self):
        """
            Opening:
        + Record Node address during unpacking
        + Calculate Node index relations based on it

            Saving
        + Update Node's own addresses
        + Update relation addresses based on index relations
        """
        self.update_sizes()

        self.update_relation_offset_in_header()

        self.update_model_pointers_offsets()

        self.update_node_offsets()

        total_mesh_and_header_size = self.file_header_size

        for mesh in self.meshes:
            total_mesh_and_header_size += mesh.total_size

        for node in self.nodes:
            if node.some_data != {}:
                total_mesh_and_header_size += 0x20

        # Node offsets
        new_node_offset = total_mesh_and_header_size

        # Calculate new offset of each node (in reverse order since that's how they are stored)
        node_size = 0x80
        for node_index in range(len(self.nodes) - 1, -1, -1):
            print("New node offset", new_node_offset)
            self.nodes[node_index].extra["Offset"] = new_node_offset
            new_node_offset += node_size

        # Go through each node and fill in new offsets
        print(len(self.nodes))
        for node in self.nodes:
            print("Updating offsets for Node", node.transform["Node Index"], "with offset", node.extra["Offset"])

            child_index = node.extra["Child Index"]
            print("Child Node index", child_index)

            if child_index != -1:
                print("Previous Child Offset", node.relation["Child Offset"])

                node.relation["Child Offset"] = self.nodes[child_index].extra["Offset"]

                print("New Child offset", node.relation["Child Offset"])

            sibling_index = node.extra["Sibling Index"]
            print("Sibling Node index", sibling_index)

            if sibling_index != -1:
                print("Previous Sibling Offset", node.relation["Sibling Offset"])

                node.relation["Sibling Offset"] = self.nodes[sibling_index].extra["Offset"]

                print("New Sibling offset", node.relation["Sibling Offset"])

    def pack_file_header(self):
        return struct.pack(self.file_header_formatting, *self.file_header.values())

    def pack_and_return(self):
        self.update_offsets()
        header_bytes = self.pack_file_header()

        mesh_bytes = b''
        for mesh in self.meshes:
            mesh_bytes += mesh.pack_and_return()

        some_data_bytes = b''
        for node in self.nodes:
            if node.some_data != {}:
                some_data_bytes += struct.pack("<8f", *node.some_data.values())

        # Nodes are stored in the reverse order of the list
        node_bytes = b''
        for node in self.nodes:
            node_bytes = node.pack_and_return() + node_bytes

        return header_bytes + mesh_bytes + some_data_bytes + node_bytes

    def save(self, file_path):
        SR2MDL_bytes = self.pack_and_return()

        new_file = open(file_path, "w+b")
        new_file.write(SR2MDL_bytes)
        new_file.close()


def meshTopology(blender_mesh) -> list:
    """
    Counts that change the moment the faces of a mesh do.

    Moving a vertex leaves all three alone, which is what makes it usable as
    "the face data on file still describes this mesh". Re-triangulating without
    touching any count would slip through, but nothing short of storing every
    index would catch that, and the file only gains from being left alone.
    """
    return [len(blender_mesh.vertices), len(blender_mesh.polygons), len(blender_mesh.loops)]


def turnSR2MeshIntoBlenderMesh(model_mesh, bl_mesh):
    temp_blender_mesh = bmesh.new()

    # Make an empty blender mesh, fill with vertices and faces
    bl_vertex_array = []

    normals = []
    uvs = []
    original_uvs = []

    for vertex in model_mesh.vertexes:
        bl_vertex = temp_blender_mesh.verts.new(vertex.position)
        bl_vertex_array.append(bl_vertex)
        normals.append(vertex.normal)
        original_uvs.append(list(vertex.uv))
        #flip V-coordinate
        flipped_v = -(vertex.uv[1] - 1.0)
        flipped_uv = [ vertex.uv[0], flipped_v ]
        uvs.append(flipped_uv)

    # Some meshes (e.g. small 4-vertex quad "dummy" props/lights) store fewer
    # than 2 face indices in the file - not enough to form even one triangle.
    # Assume the standard quad winding used by these meshes elsewhere.
    face_indices = model_mesh.faces
    if len(face_indices) < 2 and len(model_mesh.vertexes) >= 4:
        face_indices = [0, 1, 2, 2, 1, 3]

    # Stop at the last complete triangle. A file written by a tool that did not
    # triangulate its faces can leave a partial one at the end, and dropping it
    # is better than failing the whole import.
    if len(face_indices) % 3 != 0:
        print("!!! Face index count {} is not a multiple of 3, ignoring the trailing {} "
              "index(es) !!!".format(len(face_indices), len(face_indices) % 3))

    for face_index in range(0, len(face_indices) - 2, 3):
        triangle_indices = face_indices[face_index:face_index + 3]

        if max(triangle_indices) >= len(bl_vertex_array):
            print("!!! Face {} refers to a vertex outside the mesh, skipping it !!!".format(triangle_indices))
            continue

        v0 = bl_vertex_array[triangle_indices[0]]
        v1 = bl_vertex_array[triangle_indices[1]]
        v2 = bl_vertex_array[triangle_indices[2]]
        temp_blender_mesh.faces.new((v0, v1, v2))

    # Transfer all the data to bl_mesh attached to bl_obj
    temp_blender_mesh.to_mesh(bl_mesh)
    temp_blender_mesh.free()

    # If UV doesn't exist, generate new one
    channel_name = "uv0"
    try:
        bl_mesh.uv_layers[channel_name].data
    except Exception:
        bl_mesh.uv_layers.new(name=channel_name)

    for i, loop in enumerate(bl_mesh.loops):
        bl_mesh.uv_layers[channel_name].data[i].uv = uvs[loop.vertex_index]

    # Apply normals
    # Custom normals are only honoured with auto smooth on before Blender 4.1,
    # which removed the flag and always applies them
    if hasattr(bl_mesh, "use_auto_smooth"):
        bl_mesh.use_auto_smooth = True

    bl_mesh.normals_split_custom_set_from_vertices(normals)

    # Blender normalizes and quantizes custom normals, while the MDL values are
    # neither unit length nor stored at full precision. Keep the untouched
    # originals around so an unedited mesh can be written back byte for byte.
    stored_normals = bl_mesh.attributes.new(name=ORIGINAL_NORMAL_ATTRIBUTE,
                                            type='FLOAT_VECTOR',
                                            domain='POINT')
    stored_normals.data.foreach_set("vector", [value for normal in normals for value in normal])

    # Same for the UVs, which import flips and export flips back. A FLOAT_VECTOR
    # for a pair of numbers wastes a float per vertex, but it is the one type
    # this already relies on working the same way on every supported Blender
    stored_uvs = bl_mesh.attributes.new(name=ORIGINAL_UV_ATTRIBUTE,
                                        type='FLOAT_VECTOR',
                                        domain='POINT')
    stored_uvs.data.foreach_set("vector", [value for uv in original_uvs for value in (uv[0], uv[1], 0.0)])


def colorComponentToByte(component: float) -> int:
    """ A 0..1 colour component as the 0..255 byte a MDL material stores """
    return max(0, min(255, int(round(component * 255.0))))


def makeBlenderMaterial(sr2_material, material_name: str, blender_image=None):
    """
    A Blender material carrying the MDL material values.

    They live on a property group so the Material tab can edit them, see the
    add-on's SR2MDLMaterialProperties. Without the add-on registered - running
    this module on its own - the values are still kept as a custom property.
    """
    blender_material = bpy.data.materials.new(name=material_name)

    # Something recognisable in the viewport instead of the default grey
    blender_material.diffuse_color = [sr2_material[key] / 255.0 for key in MATERIAL_COLOR_0_KEYS]

    if blender_image is not None:
        sr2texture.wireImageIntoMaterial(blender_material, blender_image)

    material_properties = getattr(blender_material, MATERIAL_PROPERTY, None)

    if material_properties is None:
        blender_material["Material"] = sr2_material
        return blender_material

    material_properties.color_0 = [sr2_material[key] / 255.0 for key in MATERIAL_COLOR_0_KEYS]
    material_properties.color_1 = [sr2_material[key] / 255.0 for key in MATERIAL_COLOR_1_KEYS]

    for key in MATERIAL_FLOAT_KEYS:
        setattr(material_properties, key, sr2_material[key])

    return blender_material


def sr2MaterialFromBlenderMesh(blender_mesh, bl_object):
    """
    The MDL material to write for a mesh.

    Falls back to the "Material" custom property of objects imported before
    materials were made editable, and to the defaults for a mesh built by hand.
    """
    for blender_material in blender_mesh.materials:
        if blender_material is None:
            continue

        material_properties = getattr(blender_material, MATERIAL_PROPERTY, None)

        if material_properties is None:
            if blender_material.get("Material"):
                return dict(blender_material["Material"].to_dict())
            continue

        sr2_material = dict.copy(SR2MDL_material)

        for index, key in enumerate(MATERIAL_COLOR_0_KEYS):
            sr2_material[key] = colorComponentToByte(material_properties.color_0[index])

        for index, key in enumerate(MATERIAL_COLOR_1_KEYS):
            sr2_material[key] = colorComponentToByte(material_properties.color_1[index])

        for key in MATERIAL_FLOAT_KEYS:
            sr2_material[key] = getattr(material_properties, key)

        return sr2_material

    if bl_object.get("Material"):
        return dict(bl_object["Material"].to_dict())

    return dict.copy(SR2MDL_material)


def generate_mesh(node: SR2Node, model_mesh: Mesh, index: int, global_matrix: mathutils.Matrix,
                  model_collection, blender_image=None):
    """ Make an empty Blender object and fill it with extracted MDL data """

    # Make an empty Blender object
    node_name = 'node_{0:04}'.format(index)
    mesh_name = 'mesh_{0:04}'.format(index)

    # Make and apply custom properties to enable saving
    bl_mesh = bpy.data.meshes.new(mesh_name)

    bl_obj = bpy.data.objects.new(node_name, bl_mesh)
    bl_obj["Node Transform"] = node.transform
    bl_obj["Node Relation"] = node.relation

    bl_obj["Extra"] = node.extra

    # Attach all of this to bl_obj instead of bl_mesh to allow copying mesh from other places
    bl_obj["Model Pointers"] = model_mesh.model_pointers
    bl_obj["Draw Options"] = model_mesh.draw_options

    # Only the light billboards have one. Kept as raw 32-bit words so it can be
    # written back unchanged - see Mesh.unpack_extra_block_from_bytes
    if model_mesh.extra_block is not None:
        bl_obj[EXTRA_BLOCK_PROPERTY] = model_mesh.extra_block

    # The material goes on the mesh, where Blender expects it, so it can be
    # edited from the Material tab and follows the mesh when it gets copied
    bl_mesh.materials.append(makeBlenderMaterial(model_mesh.material,
                                                 'material_{0:04}'.format(index),
                                                 blender_image))

    model_collection.objects.link(bl_obj)

    # Select the new empty Blender object
    bpy.context.view_layer.objects.active = bl_obj
    bl_obj.select_set(True)
    bl_mesh = bpy.context.object.data

    # Apply Transforms. A node that turns out to have a parent gets this
    # redone without the axis conversion - see parentBlenderObjectsByNodeRelation
    bl_obj.matrix_local = global_matrix @ nodeTransformToMatrix(node.transform)

    turnSR2MeshIntoBlenderMesh(model_mesh, bl_mesh)

    # Recorded after the Blender mesh exists, so export can tell whether the
    # topology is still the one these bytes describe
    if model_mesh.original_face_bytes is not None:
        bl_obj[ORIGINAL_FACE_PROPERTY] = list(model_mesh.original_face_bytes)
        bl_obj[ORIGINAL_TOPOLOGY_PROPERTY] = meshTopology(bl_mesh)

    return bl_obj


def parentBlenderObjectsByNodeRelation(nodes, blender_objects):
    """
    Hang every object off the node that owns it.

    A node's relation names its first child, and that child's sibling chain
    holds the rest of them. The game draws a child through its parent's
    transform: DES_SS1/cp_4.mdl stores node_0001 at the origin, and in game it
    stands wherever node_0000 stands and turns with it. Leaving the objects
    flat put such a node at the world origin instead.

    A parented object's matrix_local is relative to its parent, which is
    exactly what a node transform holds, so the transform goes on unchanged.
    The axis conversion belongs to the root objects alone - applying it to a
    child as well would apply it once per level of the hierarchy.
    """
    already_parented = set()

    for node, bl_object in zip(nodes, blender_objects):
        child_index = node.extra["Child Index"]

        while 0 <= child_index < len(blender_objects) and child_index not in already_parented:
            already_parented.add(child_index)

            child_object = blender_objects[child_index]
            child_object.parent = bl_object

            # Without this Blender keeps the child where it is on screen by
            # remembering the offset here, and then the node transform below
            # would be applied on top of that
            child_object.matrix_parent_inverse = mathutils.Matrix()
            child_object.matrix_local = nodeTransformToMatrix(nodes[child_index].transform)

            child_index = nodes[child_index].extra["Sibling Index"]


def load(filepath: str, global_matrix: mathutils.Matrix, load_textures: bool = True):
    """
    Read a MDL into a new collection.

    global_matrix converts MDL space into Blender space. A MDL is Y-up while
    Blender is Z-up, so without one the model comes in lying on its back. It
    is stored on the collection, because export has to undo exactly the same
    conversion no matter which axes the model was brought in with.

    load_textures picks up the texture file next to the model, if there is one.
    """
    # Creating objects while in Edit Mode would leave the mesh data locked
    active_object = bpy.context.view_layer.objects.active
    if active_object is not None and active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    # Unpack the model, then make Blender object(s) out of it
    SR2_model = SR2MDL()
    SR2_model.unpack_from_file(filepath)

    # Create a collection for the model parts
    if filepath.rfind("/") == -1:
        file_name = filepath[filepath.rfind("\\") + 1:filepath.rfind(".")]
    else:
        file_name = filepath[filepath.rfind("/") + 1:filepath.rfind(".")]

    model_collection = bpy.data.collections.new(name=file_name)
    bpy.context.scene.collection.children.link(model_collection)

    # Attach file header as collection property to support saving
    model_collection["SR2MDL file header"] = SR2_model.file_header

    # Remember how the axes were converted, so export can undo it
    model_collection[AXIS_CONVERSION_PROPERTY] = [value for row in global_matrix for value in row]

    # Each node names the texture its mesh uses - see loadTexturesForModel
    blender_images = sr2texture.loadTexturesForModel(filepath) if load_textures else []

    # Turn every node into a blender object and attach mesh to it, if present
    # (nodes without a mesh can appear anywhere in the list, not just at the
    # end, so the node<->mesh association set during unpack must be used
    # instead of comparing indexes against len(SR2_model.meshes))
    blender_objects = []

    for index, node in enumerate(SR2_model.nodes):
        if node.mesh is not None:
            texture_index = node.transform["Texture Index"]

            blender_image = None
            if 0 <= texture_index < len(blender_images):
                blender_image = blender_images[texture_index]
            elif texture_index != NO_TEXTURE_INDEX and blender_images:
                print("!!! Node {} wants texture {}, which the texture file does not have !!!".format(
                    index, texture_index))

            blender_objects.append(generate_mesh(node, node.mesh, index, global_matrix,
                                                 model_collection, blender_image))
        else:
            node_name = 'node_{0:04}'.format(index)
            bl_obj = bpy.data.objects.new(node_name, None)

            # Every unpacked node has a transform, even without a mesh/some_data
            bl_obj["Node Transform"] = node.transform
            bl_obj["Extra"] = node.extra

            if node.some_data != {}:
                bl_obj["Some Data"] = node.some_data

            bl_obj["Node Relation"] = node.relation
            model_collection.objects.link(bl_obj)

            # These carry a position and scale just like the mesh nodes do -
            # leaving the empty at the origin makes export write that back.
            # Applied to matrix_local rather than matrix_world, which reads
            # back as identity until the depsgraph has caught up.
            bl_obj.matrix_local = global_matrix @ nodeTransformToMatrix(node.transform)

            blender_objects.append(bl_obj)

    # Done once every object exists, because a node can name a child that
    # comes later in the list
    parentBlenderObjectsByNodeRelation(SR2_model.nodes, blender_objects)


def isSR2MDLcollection(blender_collection):
    if isinstance(blender_collection, bpy.types.Collection):
        # print(blender_collection.name + " is a collection")
        try:
            blender_collection["SR2MDL file header"]
            return True
        except:
            return False
    else:
        return False


def collectSR2Collections(base_collection):
    collection_children = base_collection.children

    sr2_model_collections = []

    for collection in collection_children:
        if isSR2MDLcollection(collection):
            print(collection.name + " is a SR2MDL collection")
            sr2_model_collections.append(collection)

    return sr2_model_collections


def getCornerNormals(blender_mesh):
    """
    Per-corner (split) normals of a mesh.

    These are what carry the custom normals applied on import
    (normals_split_custom_set_from_vertices), unlike Mesh.vertices[].normal
    which is always recomputed by Blender from the face topology.

    Blender 4.1 removed MeshLoop.normal and Mesh.calc_normals_split() in
    favour of Mesh.corner_normals, so both spellings are supported here.
    """
    corner_normals = blender_mesh.corner_normals

    if len(corner_normals) == len(blender_mesh.loops):
        return [mathutils.Vector(normal.vector) for normal in corner_normals]

    # Blender 4.0 and older have corner_normals too, but leave it empty until
    # the split normals are calculated, and expose them on the loops instead
    blender_mesh.calc_normals_split()
    return [mathutils.Vector(loop.normal) for loop in blender_mesh.loops]


def normalCanBeCompared(normal) -> bool:
    """
    Whether a normal read from a MDL is one Blender can be asked about.

    Blender normalizes a custom normal, so a zero-length one comes back as
    whatever the faces say and a non-finite one comes back as a default. The
    value in the file never reaches the user in either case, which means the
    user cannot have edited it and it should go back untouched.

    Both happen. tree_a.mdl through tree_f.mdl zero the normals of four
    vertices each, and Char/A.MDL and Char/Z.MDL store 0, NaN, 0 for every
    vertex they have.
    """
    if not all(math.isfinite(component) for component in normal):
        return False

    return mathutils.Vector(normal).length > 0.0


def getOriginalNormals(blender_mesh):
    """ Normals as they were imported, or None if they aren't available """
    stored_normals = blender_mesh.attributes.get(ORIGINAL_NORMAL_ATTRIBUTE)

    if stored_normals is None or stored_normals.domain != 'POINT':
        return None

    if len(stored_normals.data) != len(blender_mesh.vertices):
        return None

    flat_normals = [0.0] * (len(blender_mesh.vertices) * 3)
    stored_normals.data.foreach_get("vector", flat_normals)

    return [flat_normals[i:i + 3] for i in range(0, len(flat_normals), 3)]


def getOriginalUVs(blender_mesh):
    """ UVs as they were read from the file, or None if they aren't available """
    stored_uvs = blender_mesh.attributes.get(ORIGINAL_UV_ATTRIBUTE)

    if stored_uvs is None or stored_uvs.domain != 'POINT':
        return None

    if len(stored_uvs.data) != len(blender_mesh.vertices):
        return None

    flat_uvs = [0.0] * (len(blender_mesh.vertices) * 3)
    stored_uvs.data.foreach_get("vector", flat_uvs)

    return [flat_uvs[i:i + 2] for i in range(0, len(flat_uvs), 3)]


def uvIsUnchanged(original_uv, new_uv) -> bool:
    """
    Whether a UV still holds what the file had.

    A coordinate that is not a number cannot be shown or edited, the same way a
    NaN normal cannot - the light billboards of a car store one in U - so it
    counts as untouched rather than as a mismatch that never resolves.
    """
    for axis in (0, 1):
        if not math.isfinite(original_uv[axis]):
            continue

        if abs(original_uv[axis] - new_uv[axis]) > UV_EDITED_THRESHOLD:
            return False

    return True


def convertBlenderFacesToSR2Faces(blender_mesh):
    """
    Flat array of triangle vertex indices.

    A MDL only stores triangles, so any quad or n-gon the user made has to be
    split up first - writing its indices as they are shifts every following
    triangle and corrupts the file. Blender maintains a triangulated view of
    the mesh for exactly this purpose.
    """
    blender_mesh.calc_loop_triangles()

    faces = []
    for triangle in blender_mesh.loop_triangles:
        faces.extend(triangle.vertices)

    return faces


def convertBlenderVertexesToSR2Vertexes(blender_mesh):
    vertexes = []

    # A MDL vertex holds a single normal, while Blender stores one per corner.
    # Average the corners sharing a vertex - after an untouched import they all
    # hold the same normal anyway, so this restores the original value.
    corner_normals = getCornerNormals(blender_mesh)
    averaged_normals = [mathutils.Vector((0.0, 0.0, 0.0)) for _ in blender_mesh.vertices]

    for corner_index, loop in enumerate(blender_mesh.loops):
        averaged_normals[loop.vertex_index] += corner_normals[corner_index]

    original_normals = getOriginalNormals(blender_mesh)

    # Position and normals
    for vertex_index in range(len(blender_mesh.vertices)):
        blender_vertex = blender_mesh.vertices[vertex_index]

        blender_normal = averaged_normals[vertex_index]
        if blender_normal.length > 0.0:
            blender_normal.normalize()
        else:
            # Loose vertex - not part of any face, so it has no corner normal
            blender_normal = mathutils.Vector(blender_vertex.normal)

        SR2_vertex = Vertex()
        SR2_vertex.position = [blender_vertex.co.x,
                               blender_vertex.co.y,
                               blender_vertex.co.z]

        # The original files store normals rounded to 3 decimals. Keeping that
        # here also cancels out the precision loss of Blender's custom normal
        # storage, which is quantized to 16 bits per corner.
        SR2_vertex.normal = [round(blender_normal[0], 3),
                             round(blender_normal[1], 3),
                             round(blender_normal[2], 3)]

        # Vertices that were not touched since the import get their original
        # normal back untranslated - normalizing an MDL normal (they are not
        # quite unit length) is enough to shift that third decimal on its own.
        if original_normals is not None:
            original_normal = mathutils.Vector(original_normals[vertex_index])

            if not normalCanBeCompared(original_normals[vertex_index]):
                # Nothing to compare against - Blender replaced this one on the
                # way in, so the file's value is the only one there ever was
                SR2_vertex.normal = list(original_normals[vertex_index])
            else:
                difference = max(abs(component)
                                 for component in (original_normal.normalized() - blender_normal))

                if difference <= NORMAL_EDITED_THRESHOLD:
                    SR2_vertex.normal = list(original_normals[vertex_index])

        vertexes.append(SR2_vertex)

    # UVs
    original_uvs = getOriginalUVs(blender_mesh)

    for i, loop in enumerate(blender_mesh.loops):
        fliped_uv = [blender_mesh.uv_layers["uv0"].data[i].uv[0], blender_mesh.uv_layers["uv0"].data[i].uv[1]]
        # flip V-coordinate
        fliped_uv[1] = -(fliped_uv[1] - 1.0)

        # An untouched UV goes back exactly as it was read. Flipping V in and
        # out again is exact on paper, but a V of 0.0 comes back as -0.0 -
        # negating what the first flip turned into a plain 1.0 - and rounding
        # through Blender's single precision storage can move the last bit.
        if original_uvs is not None and uvIsUnchanged(original_uvs[loop.vertex_index], fliped_uv):
            fliped_uv = list(original_uvs[loop.vertex_index])

        vertexes[loop.vertex_index].uv = fliped_uv

    return vertexes


def prepareSceneForSaving():
    """ Put the scene in a state where object transforms can be read back """
    # The operator needs something active to act on, and a model made only of
    # mesh-less nodes (tenkougen.mdl, pl01-pl03.mdl) never makes anything active
    active_object = bpy.context.view_layer.objects.active
    if active_object is not None and active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    # Reading matrix_local gives the last evaluated matrix, which is still the
    # identity for an object whose transform was set without the depsgraph
    # having run since. Flush it, or every such node exports as untransformed.
    bpy.context.view_layer.update()


def save(output_folder_path: str):
    """ Write every SR2MDL collection of the scene into a folder """
    # Select the base "Scene Collection" and get all child collections that have SR2MDL file header attached to them
    sr2_model_collections = collectSR2Collections(bpy.context.scene.collection)
    print("Found", len(sr2_model_collections), "SR2 Collections")

    for sr2_collection in sr2_model_collections:
        # " repack" is kept on this path so it never overwrites the original
        saveCollection(sr2_collection, output_folder_path + sr2_collection.name + ".mdl repack")


def collectionAxisConversion(sr2_collection) -> mathutils.Matrix:
    """
    The MDL-to-Blender conversion the collection was imported with.

    Collections from before this was recorded, and any built by hand, are
    treated as unconverted so they keep exporting the way they used to.
    """
    stored_matrix = sr2_collection.get(AXIS_CONVERSION_PROPERTY)

    if stored_matrix is None or len(stored_matrix) != 16:
        return mathutils.Matrix()

    values = list(stored_matrix)
    return mathutils.Matrix([values[row * 4:row * 4 + 4] for row in range(4)])


def saveCollection(sr2_collection, output_file_path: str):
    """ Collect all data from one collection, fill a SR2MDL with it and write it out """
    prepareSceneForSaving()

    # Objects sit in Blender space, node transforms have to go back into MDL space
    to_mdl_space = mathutils.Matrix.inverted(collectionAxisConversion(sr2_collection))

    SR2_model = SR2MDL()

    SR2_model.file_header = sr2_collection["SR2MDL file header"]

    # Go through each object and make a SR2Node and Mesh out of its data
    print(sr2_collection.name, "has", len(sr2_collection.objects), "objects")

    for bl_object in sr2_collection.objects:
        new_node = SR2Node()

        # Copy it, so filling in the new offsets below does not write back
        # into the Blender object's custom property
        new_node.transform = bl_object["Node Transform"].to_dict()

        # Objects imported before the Texture Index was recognised carry the
        # same bytes read as a float. Put them back the way they came in
        texture_index = new_node.transform.get("Texture Index")
        if isinstance(texture_index, float):
            new_node.transform["Texture Index"] = struct.unpack('<i', struct.pack('<f', texture_index))[0]

        # Only take the transform from Blender if the object was actually
        # moved. Re-encoding an untouched one would still change it: the
        # rotation is quantized to 16 bits per axis, and decomposing a
        # matrix returns Euler angles in a canonical range that need not be
        # the ones the file was written with.
        # A child's matrix_local is relative to its parent, the same way a node
        # transform is, so it goes back as it stands. Only a root carries the
        # axis conversion that has to be undone here
        if bl_object.parent is None:
            node_matrix = to_mdl_space @ bl_object.matrix_local
        else:
            node_matrix = bl_object.matrix_local.copy()

        if not matrixIsUnchanged(node_matrix, nodeTransformToMatrix(new_node.transform)):
            position, rotation, scale = node_matrix.decompose()

            new_node.transform["Position X"] = position[0]
            new_node.transform["Position Y"] = position[1]
            new_node.transform["Position Z"] = position[2]

            euler = rotation.to_euler('XYZ')
            new_node.transform["Rotation X"] = radiansToNodeRotation(euler[0])
            new_node.transform["Rotation Y"] = radiansToNodeRotation(euler[1])
            new_node.transform["Rotation Z"] = radiansToNodeRotation(euler[2])

            new_node.transform["Scale X"] = scale[0]
            new_node.transform["Scale Y"] = scale[1]
            new_node.transform["Scale Z"] = scale[2]

        # The two offsets used to be called Parent and Child before it was clear
        # they are the first child and the next sibling. Objects imported back
        # then still carry the old names; the values and their order are the same
        new_node.relation = dict(zip(SR2MDL_node_relation.keys(),
                                     bl_object["Node Relation"].to_dict().values()))

        if bl_object.get("Extra"):
            new_node.extra = bl_object["Extra"]

        if bl_object.get("Some Data"):
            new_node.some_data = bl_object["Some Data"]

        SR2_model.nodes.append(new_node)

        # Fill Mesh, if exist
        if bl_object.data:
            SR2_mesh = Mesh()

            blender_mesh = bl_object.data

            SR2_mesh.material = sr2MaterialFromBlenderMesh(blender_mesh, bl_object)

            SR2_mesh.vertexes = convertBlenderVertexesToSR2Vertexes(blender_mesh)
            SR2_mesh.faces = convertBlenderFacesToSR2Faces(blender_mesh)

            # Vertex indices are written as 2-byte values, so a mesh past
            # that limit cannot be represented and has to be split up
            if len(SR2_mesh.vertexes) > 0x7FFF:
                raise ValueError("{} has {} vertices, more than the {} a MDL mesh can index"
                                 .format(bl_object.name, len(SR2_mesh.vertexes), 0x7FFF))

            # Copied, not referenced. Assigning the property itself made every
            # recalculated offset and count below write straight back into the
            # Blender object, so a second export saw the first one's numbers
            # Copied, not referenced. Assigning the property itself made every
            # recalculated offset and count below write straight back into the
            # Blender object, so a second export saw the first one's numbers
            SR2_mesh.model_pointers = dict(bl_object["Model Pointers"].to_dict())
            original_face_count = SR2_mesh.model_pointers["Face Count"]

            SR2_mesh.model_pointers["Vertex Count"] = len(SR2_mesh.vertexes)
            # Counts indices, not triangles - a 525 triangle mesh stores 1575 here
            SR2_mesh.model_pointers["Face Count"] = len(SR2_mesh.faces)

            # A mesh whose faces are untouched goes back byte for byte, keeping
            # whatever Face Count and encoding the file used. Rewriting those
            # as triangles froze the game on the car light models
            original_faces = bl_object.get(ORIGINAL_FACE_PROPERTY)
            original_topology = bl_object.get(ORIGINAL_TOPOLOGY_PROPERTY)

            if (original_faces is not None and original_topology is not None
                    and list(original_topology) == meshTopology(blender_mesh)):
                SR2_mesh.original_face_bytes = bytes(list(original_faces))
                SR2_mesh.model_pointers["Face Count"] = original_face_count

            SR2_mesh.draw_options = bl_object["Draw Options"]

            # Written back between the faces and the Model Pointers, where it
            # came from. Its offset is recalculated on the way out, so it stays
            # right even when the mesh in front of it changed size
            extra_block = bl_object.get(EXTRA_BLOCK_PROPERTY)
            if extra_block is not None and len(extra_block) == 8:
                SR2_mesh.extra_block = list(extra_block)

            new_node.mesh = SR2_mesh
            SR2_model.meshes.append(SR2_mesh)

    SR2_model.save(output_file_path)
    print("Saved at " + output_file_path)