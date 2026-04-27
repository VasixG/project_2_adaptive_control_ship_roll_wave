class Geometry:
    pass


class Scene:
    pass


class _Base:
    class Trimesh(Geometry):
        pass


base = _Base()


def load(*args, **kwargs):
    raise RuntimeError("The local compatibility stub does not load 3D meshes.")

