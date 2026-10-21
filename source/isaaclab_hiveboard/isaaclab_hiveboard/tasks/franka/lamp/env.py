"""Franka lamp environment for Isaac Lab 3 and Newton."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.franka.board import use_franka_board, use_franka_board_viewer
from isaaclab_hiveboard.tasks.franka.common import use_franka_physics
from isaaclab_hiveboard.tasks.spot.lamp.env import SpotLampEnvCfg

from .configs.actions import FrankaLampActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import FrankaLampEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import FrankaLampSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class FrankaLampEnvCfg(SpotLampEnvCfg):
    scene: FrankaLampSceneCfg = FrankaLampSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: FrankaLampActionCfg = FrankaLampActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: FrankaLampEventCfg = FrankaLampEventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        use_franka_physics(self)
        use_franka_board(self, "lamp")


@configclass
class FrankaLampEnvCfg_PLAY(FrankaLampEnvCfg):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_franka_board_viewer(self)
        self.events.reset_board_root.params["pose_range"] = {
            key: (0.0, 0.0) for key in self.events.reset_board_root.params["pose_range"]
        }
