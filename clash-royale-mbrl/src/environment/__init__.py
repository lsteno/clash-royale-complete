"""
Environment package initialization.

For remote training with DreamerV3, use:
    from src.environment.embodied_env import ClashRoyaleEmbodiedEnv, RemoteBridgeV3

Legacy rlpyt/DreamerV1 imports (require rlpyt, dreamer-pytorch):
    from src.environment.remote_bridge import RemoteBridge, RemoteClashRoyaleEnv

Legacy local training imports (require cv2):
    from src.environment.emulator_env import ClashRoyaleEmulatorEnv, EmulatorConfig
"""

# DreamerV3 remote environment (lightweight dependencies)
from .embodied_env import (
    ClashRoyaleEmbodiedEnv,
    RemoteBridgeV3,
    RemoteStepV3,
    make_clash_royale_env,
)

# Shared utilities (no heavy dependencies)
from .action_utils import (
    ActionMapper,
    DeployCell,
    DEFAULT_DEPLOY_CELLS,
)

# Note: remote_bridge and emulator_env imports are not included here 
# to avoid rlpyt/dreamer-v1/cv2 dependencies.
# Import directly if needed for legacy code.
