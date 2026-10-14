"""Validate the live observation contract and evaluate a random or ONNX policy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', required=True)
    p.add_argument('--model')
    p.add_argument('--actions', choices=['sampled', 'deterministic'], default='sampled',
                   help="How ONNX actions are chosen. 'sampled' (default) uses the model's 'discrete_actions' output, which is "
                        "what training and Unity's default inference use. 'deterministic' uses 'deterministic_discrete_actions' "
                        "(argmax). NOTE: an argmax policy only equals the trained policy if the model is confident; the Stage 6 "
                        "model is a near-constant distribution, so argmax = always 'forward' (about 9% success).")
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=9001)
    p.add_argument('--base-port', type=int, default=None, help='Use a different port to evaluate while mlagents-learn is running (default 5005, which the trainer uses)')
    p.add_argument('--exit-set', type=int, choices=[0, 1, 2], default=None,
                   help='Random-exit scenes only: 0 all exits, 1 training exits, 2 held-out exits (default: leave the scene default)')
    p.add_argument('--blind-probs', help="No-model policy that ignores the camera: 'p_fwd,p_back,p_left,p_right' (default: uniform random)")
    p.add_argument('--blank-camera', action='store_true',
                   help='Model policies only: feed a constant background-coloured frame instead of the real camera frame. Approximates '
                        'a blanked Unity camera (the Unity-runtime test is the exact version).')
    p.add_argument('--output', default='Reports/random.json')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(width=256, height=256, time_scale=20, quality_level=0)
    params = EnvironmentParametersChannel()
    params.set_float_parameter('reset_seed', args.seed)
    if args.exit_set is not None:
        params.set_float_parameter('exit_set', float(args.exit_set))
    blind = None
    if args.blind_probs:
        blind = [float(v) for v in args.blind_probs.split(',')]
        assert len(blind) == 4 and abs(sum(blind) - 1) < 1e-6, '--blind-probs needs 4 probabilities summing to 1'
        assert not args.model, '--blind-probs is for policies without a model'
    stats = StatsSideChannel()
    env = UnityEnvironment(file_name=args.env, seed=args.seed, side_channels=[engine, params, stats],
                           no_graphics=False, timeout_wait=120, **({'base_port': args.base_port} if args.base_port else {}),
                           additional_args=['-batchmode', '-screen-fullscreen', '0', '-screen-width', '256', '-screen-height', '256'],
                           log_folder=str((output.parent / (output.stem + '-player-logs')).resolve()))
    rng = np.random.default_rng(args.seed)
    session = None
    if args.model:
        import onnxruntime as ort
        session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    returns, lengths, successes = [], [], []
    totals, steps = {}, {}
    saved = False
    try:
        env.reset()
        behavior = next(iter(env.behavior_specs))
        spec = env.behavior_specs[behavior]
        assert len(spec.observation_specs) == 1, spec.observation_specs
        assert spec.observation_specs[0].shape == (3, 84, 84), spec.observation_specs
        assert spec.action_spec.continuous_size == 0
        assert list(spec.action_spec.discrete_branches) == [4]
        while len(returns) < args.episodes:
            decision, terminal = env.get_steps(behavior)
            for i, agent in enumerate(terminal.agent_id):
                returns.append(totals.pop(agent, 0.0) + float(terminal.reward[i]))
                lengths.append(steps.pop(agent, 0))
                # GridWorld terminal reward is 1.09 on success and <=0.19 on timeout.
                successes.append(bool(terminal.reward[i] > .5 and not terminal.interrupted[i]))
                if len(returns) % 50 == 0:
                    print(f'{len(returns)}/{args.episodes} episodes; success {np.mean(successes):.1%}', flush=True)
            if len(decision):
                obs = decision.obs[0]
                assert np.isfinite(obs).all() and obs.min() >= 0 and obs.max() <= 1
                if not saved:
                    assert obs.std() > .01, 'Camera is blank'
                    pixels = obs[0].transpose(1, 2, 0)
                    assert np.any((pixels[:,:,2] > .7) & (pixels[:,:,0] < .4)), 'Blue agent missing'
                    assert np.any((pixels[:,:,1] > .7) & (pixels[:,:,0] < .4)), 'Green target missing'
                    Image.fromarray((pixels * 255).astype(np.uint8)).save(output.with_suffix('.png'))
                    saved = True
                if session:
                    feeds = {}
                    for item in session.get_inputs():
                        if item.name == 'obs_0':
                            frame = obs.astype(np.float32)
                            if args.blank_camera:
                                frame = np.zeros_like(frame); frame[:, 0] = .03; frame[:, 1] = .03; frame[:, 2] = .04
                            feeds[item.name] = frame
                        elif item.name == 'action_masks':
                            feeds[item.name] = np.ones((len(decision), 4), dtype=np.float32)
                        else:
                            raise RuntimeError(f'Unexpected model input: {item.name} {item.shape}')
                    out_name = 'discrete_actions' if args.actions == 'sampled' else 'deterministic_discrete_actions'
                    action = session.run([out_name], feeds)[0].astype(np.int32)
                else:
                    if blind is None:
                        action = rng.integers(0, 4, size=(len(decision), 1), dtype=np.int32)
                    else:
                        action = rng.choice(4, size=(len(decision), 1), p=blind).astype(np.int32)
                for i, agent in enumerate(decision.agent_id):
                    totals[agent] = totals.get(agent, 0.0) + float(decision.reward[i])
                    steps[agent] = steps.get(agent, 0) + 1
                env.set_actions(behavior, ActionTuple(discrete=action))
            env.step()
            stats.get_and_reset_stats()
        n = len(successes)
        rate = float(np.mean(successes))
        z = 1.96
        center = (rate + z*z/(2*n))/(1+z*z/n)
        radius = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        report = dict(policy=args.model or 'random', action_selection=(args.actions if args.model else ('blind ' + args.blind_probs if blind else 'uniform random')), exit_set=args.exit_set, blank_camera=args.blank_camera, seed=args.seed, episodes=n,
                      model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None,
                      observation_shape=[3,84,84], observation_layout='CHW', discrete_branches=[4],
                      successes=int(sum(successes)), success_rate=rate,
                      success_rate_95ci=[center-radius, center+radius],
                      mean_return=float(np.mean(returns)), mean_length=float(np.mean(lengths)),
                      mean_success_length=float(np.mean([l for l,s in zip(lengths,successes) if s])) if any(successes) else None,
                      episode_returns=returns, episode_lengths=lengths, episode_success=successes)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if not k.startswith('episode_')}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
