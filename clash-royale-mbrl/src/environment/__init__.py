"""
Environment package initialization.

For remote training with DreamerV3, use:
    from src.environment.embodied_env import ClashRoyaleEmbodiedEnv, RemoteBridgeV3

For local Redroid hive training with DreamerV3, use:
    from src.environment.hive_env import ClashRoyaleHiveEmbodiedEnv, HiveEnvConfig

Legacy rlpyt/DreamerV1 imports (require rlpyt, dreamer-pytorch):
    from src.environment.remote_bridge import RemoteBridge, RemoteClashRoyaleEnv

Legacy local training imports (require cv2):
    from src.environment.emulator_env import ClashRoyaleEmulatorEnv, EmulatorConfig
"""

# Keep this package import-light.
#
# Importing `embodied_env` pulls in DreamerV3 dependencies (e.g. `elements`),
# which are not required for perception-only services like the gRPC
# FrameService. Downstream code should import from the specific module it
# needs, e.g. `from src.environment.embodied_env import RemoteBridgeV3`.

# Shared utilities (no heavy dependencies)
from .action_utils import (
    ActionMapper,
    DeployCell,
    DEFAULT_DEPLOY_CELLS,
)

# Note: remote_bridge and emulator_env imports are not included here 
# to avoid rlpyt/dreamer-v1/cv2 dependencies.
# Import directly if needed for legacy code.
