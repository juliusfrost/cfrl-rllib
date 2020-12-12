import models.minigrid


def register():
    """
    Registers all models as available for RLlib
    """
    models.minigrid.register()
