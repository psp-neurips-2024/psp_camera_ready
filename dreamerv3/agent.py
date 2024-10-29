import functools

import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train', include_recon=False, v_expl_mode='none'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    image_v_grad = None
    v = None
    if v_expl_mode == 'none':
      embed = self.wm.encoder(obs)
      latent, _ = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, obs['is_first'])
    elif (v_expl_mode == 'gradient'
          or v_expl_mode == 'gradient_x_intensity'
          or v_expl_mode == 'integrated_gradient'):
      vf = self._gradient_weighting_nets
      def get_v_v_latent(d_data, data, state):
        (prev_latent, prev_action), _, _ = state
        all_data = {**d_data, **data}
        embed = self.wm.encoder(all_data)
        # Return value from obs_step is post, prior
        latent, _ = self.wm.rssm.obs_step(
          prev_latent, prev_action, embed, all_data['is_first'])
        v = 0.
        for vf_ in vf:
          v += vf_(latent).mean()
        return v, (v, latent)
      get_v_v_latent = jax.jacrev(get_v_v_latent, has_aux=True)

      if (v_expl_mode == 'gradient'
          or v_expl_mode == 'gradient_x_intensity'):
        image_v_grad, (v, latent) = get_v_v_latent(
          {'image': obs['image']},
          {k: v for k, v in obs.items() if k != 'image'},
          state
        )
        batch_length = image_v_grad['image'].shape[0]
        image_v_grad = tree_map(
          lambda x: x[jnp.arange(batch_length), jnp.arange(batch_length)],
          image_v_grad
        )
        image_v_grad = image_v_grad['image']
        if v_expl_mode == 'gradient_x_intensity':
          image_v_grad *= obs['image']
      else:
        alphas = jnp.linspace(0, 1, 11)[:, None, None, None, None]
        images = obs['image'][None, ...]
        interp_images = alphas * images
        interp_images = {'image': interp_images}
        other_items = {k: jnp.repeat(v[None, ...], 11, axis=0) for k, v in obs.items() if k != 'image'}
        state = tree_map(lambda x: jnp.repeat(x[None, ...], 11, axis=0), state)
        get_v_v_latent = jax.vmap(get_v_v_latent)
        image_v_grad, (v, latent) = get_v_v_latent(
          interp_images,
          other_items,
          state
        )
        batch_length = image_v_grad['image'].shape[1]
        image_v_grad = tree_map(
          lambda x: x[:, jnp.arange(batch_length), jnp.arange(batch_length)],
          image_v_grad
        )
        image_v_grad = image_v_grad['image']
        image_v_grad = jnp.mean(image_v_grad, axis=0) * obs['image']
        v = v[-1]
        latent = tree_map(lambda x: x[-1], latent)
      # Note: Not quite equivalent to train because we're not calling
      #       absolute within policy, instead we do that in the caller.
      image_v_grad = (
          jnp.ones_like(image_v_grad) * (
              1 - self.config.image_v_grad_interp_value)
          + image_v_grad * self.config.image_v_grad_interp_value)
    else:
      raise ValueError(f'v_expl_mode {v_expl_mode} not supported.')

    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    outs = None
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())

    if include_recon:
      if outs is None:
        outs = {}
      outs['recon'] = self.wm.heads['decoder'](latent)['image'].mode()

    if v_expl_mode != 'none':
      outs['v'] = v
      outs['image_expl'] = image_v_grad

    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  @property
  def _gradient_weighting_nets(self):
    available_nets = {
      'value_function': self.task_behavior.ac.critics['extr'].net,
      'reward_function': self.wm.heads['reward'],
      'policy_function': self.task_behavior.ac.actor,
    }
    return [available_nets[net] for net in self.config.gradient_weighting_nets]

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    vf = self._gradient_weighting_nets
    state, wm_outs, mets = self.wm.train(data, state, vf)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    vf = self._gradient_weighting_nets
    report.update(self.wm.report(data, vf))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      elif key != 'masks'and key != 'masks_count':
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    obs['step'] = jnp.array([int(self.step)] * len(obs['image']))
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(
      shapes, **config.encoder,
      train_image_augmentation=config.image_augmentation.enabled,
      train_image_augmentation_mean=config.image_augmentation.mean,
      train_image_augmentation_std=config.image_augmentation.std,
      name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    decoder_shapes = shapes.copy()
    decoder_config = config.decoder
    if self.config.adversarial_action_head:
      decoder_shapes['action'] = tuple(self.act_space.shape)
      # Make this less hacky.
      decoder_config = decoder_config.update({'mlp_keys': 'action'})
    self.heads = {
        'decoder': nets.MultiDecoder(
          decoder_shapes, **decoder_config, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    if self.config.embed_only_action_adversarial_head:
      decoder_config = decoder_config.update({
        'inputs': ['embed'], 'mlp_keys': 'action', 'cnn_keys': 'sfafdaf'})
      self.heads['embed_action_head'] = nets.MultiDecoder(
          {'action': self.act_space.shape},
          name='embed_dec',
          **decoder_config
      )
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state, vf):
    modules = [self.encoder, self.rssm, *self.heads.values()]

    adv_modules = None
    adv_lossfn = None
    if (self.config.adversarial_action_head
        or self.config.embed_only_action_adversarial_head):
      adv_modules = [self.encoder, self.rssm]
      adv_lossfn = functools.partial(self.loss, act_adv_round=True)
    mets, (state, outs, _, metrics) = self.opt(
        modules, self.loss, data, state, vf, has_aux=True,
        adv_modules=adv_modules, adv_lossfn=adv_lossfn)
    metrics.update(mets)
    return state, outs, metrics

  def get_embed_post_prior(self, d_data, data, state):
    all_data = {}
    all_data.update(d_data)
    all_data.update(data)
    embed = self.encoder(all_data, training=True)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
      prev_action[None], all_data['action'][:-1]], 0)
    post, prior = self.rssm.observe(
      embed, prev_actions, all_data['is_first'], prev_latent,
      batch_free=True)
    return post, (embed, prior)

  def get_v_embed_post_prior(self, d_data, data, state, vf=None):
    post, (embed, prior) = self.get_embed_post_prior(d_data, data, state)
    vf_post_mean = None
    if vf:
      vf_post_mean = 0.
      for vf_ in vf:
        vf_post_mean_ = vf_(post).mean()
        ndims = vf_post_mean_.ndim
        if ndims != 1:
          vf_post_mean_ = jnp.mean(vf_post_mean_, axis=tuple(range(1, ndims)))
        vf_post_mean += vf_post_mean_
    return vf_post_mean if vf is not None else None, (embed, post, prior,)

  def per_item_loss(self, data, state, vf, act_adv_round=False):
    """
    World Model loss

    :param data:
    :param state:
    :param vf: vf. If passed in should be a function that takes an array
      with last dimension representing posterior from world model and
      outputs value. Will be used to take the derivative of value
      function with regard to each state and weight loss terms of state
      prediction according to their contribution.
    :return:
    """
    embed = post = prior = image_v_grad = latent_v_grad = None
    if (not self.config.image_v_grad
        and not self.config.dyn_v_grad
        and not self.config.rep_v_grad):
      _, (embed, post, prior) = self.get_v_embed_post_prior({}, data, state)
    else:
      if self.config.image_v_grad:
        embed_post_prior = jax.jacrev(
            functools.partial(self.get_v_embed_post_prior, vf=vf), has_aux=True)

        d_data = {}
        for k in self.encoder.cnn_shapes:
          if k in data:
            d_data[k] = data[k]
            del data[k]

        image_v_grad, (embed, post, prior) = embed_post_prior(d_data, data, state)
        batch_length = image_v_grad['image'].shape[0]
        if self.config.image_v_grad_backprop_truncation > 1:
          window = self.config.image_v_grad_backprop_truncation
          coords = jnp.arange(batch_length)
          dist = coords[:, None] - coords[None, :]
          attends = (dist < window) & (dist >= 0)
          image_v_grad = tree_map(
              lambda x: ((
                  (x * attends[..., None, None, None]).astype(
                    jnp.float32).sum(axis=0)
                  / attends[..., None, None, None].astype(
                    jnp.float32).sum(axis=0))).astype(x.dtype),
              image_v_grad)
        else:
          image_v_grad = tree_map(
              lambda x: x[jnp.arange(batch_length), jnp.arange(batch_length)],
              image_v_grad)
        image_v_grad = sg(image_v_grad)

        data.update(d_data)
      if self.config.dyn_v_grad or self.config.rep_v_grad:
        if not self.config.image_v_grad:
          post, (embed, prior) = self.get_embed_post_prior({}, data, state)
        v_mean = jax.jacrev(lambda post: vf(post).mean())
        latent_v_grad = v_mean(post)
        batch_length = latent_v_grad['deter'].shape[0]
        latent_v_grad = tree_map(
            lambda x: x[jnp.arange(batch_length), jnp.arange(batch_length)],
            latent_v_grad)
        latent_v_grad = sg(latent_v_grad)

    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      if isinstance(head, nets.MultiDecoder):
        out = head(
          feats if name in self.config.grad_heads else sg(feats),
          act_adv_round=act_adv_round)
      else:
       out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    rssm_loss_weight = None
    if self.config.dyn_v_grad or self.config.rep_v_grad:
      rssm_loss_weight = jax.lax.cond(
        data['step'] >= self.config.v_grad_warmup_steps,
        lambda x: x,
        lambda x: tree_map(jnp.ones_like, x),
        latent_v_grad
      )
    if not act_adv_round:
      losses['dyn'] = self.rssm.dyn_loss(
          post, prior, **self.config.dyn_loss,
          weight=rssm_loss_weight if self.config.dyn_v_grad else None,
          normed=self.config.latent_v_grad_normed,
          keep_magnitude=self.config.latent_v_grad_norm_keep_magnitude,
          percentile_clip=self.config.latent_v_grad_percentile_clip)
      losses['rep'] = self.rssm.rep_loss(
          post, prior, **self.config.rep_loss,
          weight=rssm_loss_weight if self.config.rep_v_grad else None,
          normed=self.config.latent_v_grad_normed,
          keep_magnitude=self.config.latent_v_grad_norm_keep_magnitude,
          percentile_clip=self.config.latent_v_grad_percentile_clip)
    for key, dist in dists.items():
      if image_v_grad is not None and key in self.encoder.cnn_shapes:
        # Get absolute value of gradients of value (or reward) wrt each image
        # pixel/channel, plus a small value epsilon in some cases, as initial
        # version of weights.
        weights = jnp.abs(image_v_grad[key])  # [L, H, W, C]
        if self.config.image_v_grad_normed:
          weights += 1e-6
        if self.config.image_v_grad_x_intensity:
          weights *= data[key]

        # masks_count = data['masks_count']  # [L,]
        if self.config.image_v_grad_mask_level:
          masks = data['masks']  # [L, M, H, W]
          neg_mask = ~jnp.any(masks, axis=1, keepdims=True)  # [L, 1, H, W]
          masks = jnp.concatenate([masks, neg_mask], axis=1)
          p99 = jnp.percentile(weights, 99)
          numerator = jnp.clip(weights, 0, p99)
          weights_before_mask_aggregation = numerator
          numerator = (  # [L, M, H, W, C]
            numerator[:, None, ...]  # [L, 1, H, W, C]
            * masks[..., None]  # [L, M, H, W, 1]
          )
          mask_scores = (  # [L, M]
            numerator.sum((-1, -2, -3))  # [L, M, H, W, C] -> [L, M]
            / masks.sum((-1, -2))  # [L, M, H, W] -> [L, M]
          )
          mask_scores = jnp.nan_to_num(mask_scores)  # [L, M]
          weights = mask_scores[..., None, None] * masks  # [L, M, H, W]
          weights = weights.sum(axis=1)  # [L, H, W]
          weights = weights[..., None]  # [L, H, W, 1], last dim replaces C.

        if self.config.image_v_grad_interp_value != 1.:
          # Normalizing each frame's weights to sum to 1.
          original_weight_sum = jnp.prod(jnp.array(weights.shape[-3:]))
          weights = (
              weights  # [L, H, W, C]
              * (  # [L, 1, 1, 1]
                  original_weight_sum  # [1,]
                  / weights.sum((-1, -2, -3), keepdims=True)  # [L, 1, 1, 1]
              )
          )

          # Any frame that has no weights at all is set to a uniform weighting.
          frame_nans = jnp.any(
              ~jnp.isfinite(weights), (-1, -2, -3), keepdims=True)  # [L, 1, 1, 1]
          weights = jnp.where(frame_nans, 1., weights)

          # Interpolating between gradient weighted reconstruction loss and
          # a uniform weighting.
          weights = (
              jnp.ones_like(weights) * (
                  1 - self.config.image_v_grad_interp_value
              )
              + weights * self.config.image_v_grad_interp_value
          )

        if self.config.image_v_grad_percentile_clip:
          p95 = jnp.percentile(weights, 95)
          weights = jnp.where(weights > p95, p95, weights)

        if self.config.image_v_grad_normed:
          weights /= jnp.sum(weights, axis=-1, keepdims=True)

        warmup = self.config.v_grad_warmup_steps
        if warmup > 0:
          weights_scale = jnp.minimum(data['step'] / warmup, 1)
        else:
          weights_scale = jnp.ones_like(data['step'])

        weights = (
            jnp.ones_like(weights) * (1 - weights_scale)
            + weights * weights_scale)

        pred = dist.mean()
        loss = (pred - data[key].astype(jnp.float32)) ** 2

        original_loss = loss
        loss = weights * loss

        loss = loss.sum(
            # (C, W, H)
            (-1, -2, -3)
        )
      else:
        if key == 'action':
          prev_latent, prev_action = state
          truth = jnp.concatenate([
              prev_action[None], data['action'][:-1]], 0)
        else:
          truth = data[key]
        loss = -dist.log_prob(truth.astype(jnp.float32))
      assert loss.shape == embed.shape[:1], (key, loss.shape)
      if not act_adv_round or key == 'action':
        losses[key] = loss
        if key == 'action':
         losses[key] *= self.config.adversarial_action_head_scale
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed': embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[-1] for k, v in post.items()}
    last_action = data['action'][-1]
    state = last_latent, last_action
    # TODO: Convert dists to tensors and return.

    gradients_dict = {}
    if self.config.dyn_v_grad or self.config.rep_v_grad:
      gradients_dict['latent_weight'] = latent_v_grad['stoch']
    if self.config.image_v_grad:
      image_v_grad = image_v_grad['image']
      if self.config.image_v_grad_x_intensity:
        image_v_grad *= data['image']
      gradients_dict['image_weight'] = image_v_grad
      gradients_dict['scaled_image_weight'] = weights
      if self.config.image_v_grad_mask_level:
        gradients_dict['preagg_image_weight'] = weights_before_mask_aggregation
      gradients_dict['image_weight_scale_factor'] = weights_scale
    return (data, post, prior, losses, model_loss, state, out,
            gradients_dict)

  def loss(self, data, state, vf, act_adv_round=False):
    """
    World Model loss

    :param data:
    :param state:
    :param vf: vf. If passed in should be a function that takes an array
      with last dimension representing posterior from world model and
      outputs value. Will be used to take the derivative of value
      function with regard to each state and weight loss terms of state
      prediction according to their contribution.
    :return:
    """
    per_item_loss = jax.vmap(
        functools.partial(
          self.per_item_loss, vf=vf, act_adv_round=act_adv_round),
          in_axes=[0, 0])
    (data, post, prior, losses, model_loss, state, out,
     gradients_dict) = per_item_loss(data, state)

    metrics = self._metrics(data, post, prior, losses, model_loss, gradients_dict)
    return model_loss.mean(), (state, out, gradients_dict, metrics,)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data, vf):
    state = self.initial(len(data['is_first']))
    report = {}
    _, (_, _, gradients_dict, metrics) = self.loss(data, state, vf)
    report.update(metrics)
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      if 'scaled_image_weight' in gradients_dict:
        scaled = (
          gradients_dict['scaled_image_weight'][:6].sum(axis=-1)
        )
        weighting = jnp.zeros_like(scaled, shape=scaled.shape + (3,))
        weighting = weighting.at[..., 0].set(scaled)
        weighting = (weighting - weighting.min()) / (weighting.max() - weighting.min())
        video = jnp.concatenate([video, weighting], axis=2)
      if 'preagg_image_weight' in gradients_dict:
        scaled = (
          gradients_dict['preagg_image_weight'][:6].sum(axis=-1)
        )
        weighting = jnp.zeros_like(scaled, shape=scaled.shape + (3,))
        weighting = weighting.at[..., 0].set(scaled)
        weighting = (weighting - weighting.min()) / (weighting.max() - weighting.min())
        video = jnp.concatenate([video, weighting], axis=2)
      if 'masks' in data:
        masks = data['masks'][:6]  # [B, L, M, H, W]
        colors = jax.random.uniform(
            nj.rng(), jnp.shape(masks)[:3] + (3,))  # [B, L, M, C]
        masks = masks[..., None]  # [B, L, M, H, W, 1]
        masks = masks * colors[:, :, :, None, None, :]  # [B, L, M, H, W, C]
        masks = masks.sum(axis=2)  # [B, L, H, W, C]
        video = jnp.concatenate([video, masks], axis=2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, post, prior, losses, model_loss, gradients_dict):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    for k, v in gradients_dict.items():
      metrics.update(jaxutils.tensorstats(v, k))
    # TODO: Re-enable dists stats. Requires passing dist parameters through
    #       vectorized per-item-loss fn which is non-trivial.
    # metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    # if 'reward' in dists and not self.config.jax.debug_nans:
    #   stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
    #   metrics.update({f'reward_{k}': v for k, v in stats.items()})
    # if 'cont' in dists and not self.config.jax.debug_nans:
    #   stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
    #   metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
