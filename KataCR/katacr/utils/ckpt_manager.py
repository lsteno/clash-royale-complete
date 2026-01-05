import orbax.checkpoint as ocp
from pathlib import Path
import shutil
from flax.training import train_state


class CheckpointManager(ocp.CheckpointManager):
  """Thin wrapper matching older Orbax API used by KataCR checkpoints."""

  def __init__(self, path_save, max_to_keep=1, remove_old=False):
    self.path_save = path_save = str(Path(path_save).resolve())
    if remove_old:
      shutil.rmtree(path_save, ignore_errors=True)

    checkpointers = {
      'variables': ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
      'config': ocp.Checkpointer(ocp.JsonCheckpointHandler()),
    }
    super().__init__(
      path_save,
      checkpointers=checkpointers,
      options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, step_format_fixed_length=3),
    )

  def save(self, epoch: int, state: train_state.TrainState, config: dict, verbose: bool = True):
    # Normalize paths to strings for JSON serialization.
    for k, v in list(config.items()):
      if isinstance(v, Path):
        config[k] = str(v)
    config['_step'] = int(state.step)
    if verbose:
      print(f"Save weights at {self.path_save}/{epoch:03}/")

    variables = {'params': state.params}
    if hasattr(state, 'batch_stats'):
      variables['batch_stats'] = state.batch_stats

    items = {
      'variables': variables,
      'config': config,
    }
    return super().save(epoch, items)

  def restore(self, epoch: int):
    return super().restore(epoch)

  def load(self, state: train_state.TrainState, epoch: int, need_opt: bool = False):
    ret = self.restore(epoch)
    if not isinstance(ret, dict):
      return state
    cfg = ret.get('config', {}) or {}
    params = ret.get('variables', {}).get('params', None)
    batch_stats = ret.get('variables', {}).get('batch_stats', None)
    if params is not None:
      state = state.replace(params=params)
    if batch_stats is not None and hasattr(state, 'batch_stats'):
      state = state.replace(batch_stats=batch_stats)
    if '_step' in cfg:
      state = state.replace(step=cfg['_step'])
    return state
