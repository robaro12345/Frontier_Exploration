"""Check persisted measured metrics, failed attempts, summaries and exported plots."""

import json

from fronter_exploration.RL.reporting import (
    combine_reports, discover_reports, main, read_records, summarize)
from fronter_exploration.RL.training import PARAMETERS, Training
import numpy as np
import pytest
from test_rl import Logger
from test_rl_integration import orchestration_node


def trainer(directory, episodes=3):
    """Use real replay and optimizer updates for the reporting integration."""
    return Training(dict(PARAMETERS, batch_size=1, episode_limit=episodes,
                         checkpoint_dir=str(directory)), Logger())


def test_graphs_only_at_end_of_run_and_exported_once(tmp_path, monkeypatch):
    """Intermediate episode reports must not create or refresh training figures."""
    import fronter_exploration.RL.training as module
    original = module.plot_records
    exports = []

    def export(records, directory):
        exports.append(len(records))
        return original(records, directory)

    monkeypatch.setattr(module, 'plot_records', export)
    training = trainer(tmp_path, episodes=2)
    state = {3: np.array([.2, .4], dtype=np.float32)}
    try:
        training.start_episode(.5)
        training.decide({}, (state, 3, 1., {}, True),
                        {'known_fraction': .7}, terminal=True).result(5)
        assert not training.finished and exports == []
        assert (training.report.directory / 'episodes.csv').is_file()
        assert not list(training.report.directory.glob('*.png'))
        assert not list(training.report.directory.glob('*.pdf'))

        training.start_episode(.5)
        training.decide({}, (state, 3, 2., {}, True),
                        {'known_fraction': .8}, terminal=True).result(5)
        assert training.finished and exports == [2]
        assert len(list(training.report.directory.glob('*.png'))) == 4
        assert len(list(training.report.directory.glob('*.pdf'))) == 4
        training.close()
        assert exports == [2]  # Cleanup must not export the final graphs again.
    finally:
        training.close()


def test_actual_metrics_failed_attempt_summary_and_graph_data(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    observed = []
    original = Axes.plot

    def observe(axis, x, y, *args, **kwargs):
        observed.append((list(x), list(y)))
        return original(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', observe)
    training = trainer(tmp_path)
    state = {3: np.array([.2, .4], dtype=np.float32)}
    training.prepare_episode('map01_open_obstacles.sdf', 0, (3.05, 8.55, -1.54))
    training.decide(state, (state, 3, 1.25, state, False),
                    {'navigation_success': True, 'known_fraction': .7}).result(5)
    training.decide({}, (state, 3, -.5, {}, True),
                    {'navigation_success': False, 'known_fraction': .8}, terminal=True).result(5)
    training.prepare_episode('map02_corridor_loop.sdf', 1, (1., 2., 3.))
    training.decide(state, (state, 3, -2., state, False),
                    {'navigation_success': False, 'known_fraction': .6}).result(5)
    training.abort_episode('No fresh post-map', coverage=.65).result(5)
    training.close()
    directory = training.report.directory
    records = read_records(directory / 'episodes.csv')
    assert len(records) == 2 and training.completed_episodes == training.agent.episodes == 1
    assert records[0]['cumulative_reward'] == .75 and records[0]['steps'] == 2
    assert records[0]['coverage'] == .8 and records[0]['success']
    assert records[0]['navigation_successes'] == 1
    assert records[0]['world'] == 'map01_open_obstacles.sdf'
    assert records[1]['cumulative_reward'] == -2. and records[1]['steps'] == 1
    assert records[1]['coverage'] == .65 and not records[1]['success']
    assert records[1]['outcome'] == 'aborted' and records[1]['reason'] == 'No fresh post-map'
    assert records[1]['optimizer_updates'] == training.agent.updates == 3
    assert records[1]['epsilon'] == training.agent.epsilon == pytest.approx(.995 ** 3)
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary == summarize(records, training.agent.epsilon)
    assert summary['total_episodes'] == 2 and summary['success_rate'] == .5
    assert summary['mean_episode_reward'] == -.625
    assert summary['mean_steps_per_episode'] == 1.5
    text = (directory / 'training_report.txt').read_text()
    assert 'Episode: 1\nCumulative Reward: 0.750000\nSteps: 2\nCoverage: 80.00%' in text
    assert 'Episode: 2\nCumulative Reward: -2.000000\nSteps: 1\nCoverage: 65.00%' in text
    assert 'Success: False' in text and 'Success Rate: 50.00%' in text
    assert ([1, 2], [.75, -2.]) in observed
    assert ([1, 2], [100., 50.]) in observed
    assert ([1, 2], [2, 1]) in observed
    for name in ('reward_vs_episode', 'success_rate_vs_episode',
                 'steps_per_episode', 'training_curves'):
        assert (directory / (name + '.png')).read_bytes().startswith(b'\x89PNG')
        assert (directory / (name + '.pdf')).read_bytes().startswith(b'%PDF')
    count = len(training.report.records)
    training.close()  # Repeated cleanup must not create another failed attempt.
    assert len(training.report.records) == count


def test_interrupted_startup_has_no_invented_coverage_or_steps(tmp_path):
    training = trainer(tmp_path)
    training.prepare_episode('map01_open_obstacles.sdf', 0, (3.05, 8.55, -1.54))
    training.close(reason='Interrupted during simulator startup')
    row = read_records(training.report.directory / 'episodes.csv')[0]
    assert row['coverage'] is None and row['steps'] == 0
    assert row['cumulative_reward'] == 0. and row['epsilon'] == 1.
    assert row['outcome'] == 'interrupted' and not row['success']
    assert 'Coverage: N/A' in (training.report.directory / 'training_report.txt').read_text()
    assert not (tmp_path / 'latest.pt').exists()


def test_empty_run_and_resume_keep_reports_separate(tmp_path):
    first = trainer(tmp_path)
    first.close()
    summary = json.loads((first.report.directory / 'summary.json').read_text())
    assert summary['total_episodes'] == 0
    assert summary['mean_episode_reward'] is None and summary['success_rate'] is None
    assert summary['mean_steps_per_episode'] is None and summary['final_epsilon'] == 1.
    assert not list(first.report.directory.glob('*.png'))
    second = trainer(tmp_path)
    second.close()
    assert first.report.directory != second.report.directory
    assert (first.report.directory / 'summary.json').is_file()


def test_main_abort_records_failure_without_false_completion(tmp_path):
    node = orchestration_node()
    node.rl = node.training = True
    node.trainer = trainer(tmp_path)
    node.trainer.start_episode(.5)
    node._abort_rl('Navigation transport lost')
    node._rl_future.result(5)
    node._poll_rl()
    assert node.episode_aborted and node._rl_future is None
    assert node.trainer.completed_episodes == 0 and node.trainer.agent.episodes == 0
    node.trainer.close()
    row = node.trainer.report.records[0]
    assert not row['success'] and row['reason'] == 'Navigation transport lost'
    assert row['coverage'] == .5


def test_cli_regenerates_exported_figures_from_saved_csv(tmp_path, capsys):
    training = trainer(tmp_path)
    training.prepare_episode('map01_open_obstacles.sdf', 0, (3.05, 8.55, -1.54))
    training.abort_episode('Test abort', coverage=.5).result(5)
    training.close()
    output = tmp_path / 'exported'
    main([str(training.report.directory / 'episodes.csv'), '--output-dir', str(output)])
    assert len(list(output.glob('*.png'))) == len(list(output.glob('*.pdf'))) == 4
    assert str(output / 'reward_vs_episode.png') in capsys.readouterr().out


def test_combined_resume_report_retains_failures_and_measured_curves(tmp_path, monkeypatch):
    """Merge actual worker records across a checkpoint resume without losing history."""
    state = {3: np.array([.2, .4], dtype=np.float32)}
    first = trainer(tmp_path)
    first.prepare_episode('map01.sdf', 0, (1., 2., 3.))
    first.decide({}, (state, 3, 1.25, {}, True), terminal=True).result(5)
    first.prepare_episode('map02.sdf', 9, (-6.15, -2.05, -2.14))
    first.abort_episode('Startup timed out', coverage=.01).result(5)
    first.close()
    second = Training(dict(PARAMETERS, batch_size=1, episode_limit=1,
                           checkpoint_dir=str(tmp_path),
                           checkpoint=str(tmp_path / 'latest.pt')), Logger())
    second.prepare_episode('map02.sdf', 9, (-6.15, -2.05, -2.14))
    second.decide({}, (state, 3, 2.5, {}, True), terminal=True).result(5)
    second.close()
    sources = [first.report.directory / 'episodes.csv', second.report.directory / 'episodes.csv']
    before = {path: path.read_bytes() for report in (first.report, second.report)
              for path in report.directory.iterdir()}
    from matplotlib.axes import Axes
    plotted = []
    original = Axes.plot

    def observe(axis, x, y, *args, **kwargs):
        plotted.append((list(x), list(y)))
        return original(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', observe)
    output = tmp_path / 'combined'
    main([str(path) for path in sources] + ['--output-dir', str(output)])
    rows = read_records(output / 'episodes.csv')
    assert [row['episode'] for row in rows] == [1, 2, 3]
    assert [row['outcome'] for row in rows] == ['completed', 'aborted', 'completed']
    assert rows[1]['reason'] == 'Startup timed out'
    assert rows[1]['world'] == rows[2]['world'] == 'map02.sdf'
    assert rows[1]['pose_index'] == rows[2]['pose_index'] == '9'
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['total_episodes'] == 3 and summary['completed_episodes'] == 2
    assert summary['mean_episode_reward'] == 1.25
    assert summary['mean_steps_per_episode'] == pytest.approx(2 / 3)
    assert summary['success_rate'] == pytest.approx(2 / 3)
    assert summary['final_epsilon'] == second.agent.epsilon == pytest.approx(.995 ** 2)
    assert ([1, 2, 3], [1.25, 0., 2.5]) in plotted
    assert ([1, 2, 3], [100., 50., 200 / 3]) in plotted
    assert ([1, 2, 3], [1, 0, 1]) in plotted
    assert 'Episode: 3' in (output / 'training_report.txt').read_text()
    assert json.loads((output / 'sources.json').read_text()) == [str(path) for path in sources]
    assert len(list(output.glob('*.png'))) == len(list(output.glob('*.pdf'))) == 4
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize('duplicate', [True, False])
def test_combined_report_rejects_duplicate_sources_and_source_overwrite(tmp_path, duplicate):
    """Reject double counting and overwriting a source before touching any file."""
    source = tmp_path / 'source' / 'episodes.csv'
    paths = [source, source] if duplicate else [source]
    output = tmp_path / 'combined' if duplicate else source.parent
    with pytest.raises(ValueError, match='twice|separate'):
        combine_reports(paths, output)
    assert not output.exists()


@pytest.mark.parametrize('with_output', [True, False])
def test_combined_cli_missing_inputs_preserves_output(tmp_path, capsys, with_output):
    """Explain missing history or output paths without writing a partial report."""
    output = tmp_path / 'combined'
    args = [str(tmp_path / 'missing.csv'), str(tmp_path / 'also_missing.csv')]
    if with_output:
        args += ['--output-dir', str(output)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and not output.exists()
    message = capsys.readouterr().err
    assert ('No such file' if with_output else '--output-dir is required') in message


def test_all_runs_cli_preserves_order_failures_and_excludes_merged_outputs(tmp_path):
    """Discover raw/recovered reports; repeat export without importing its own output."""
    import shutil
    run = trainer(tmp_path / 'scratch')
    run.prepare_episode('map01.sdf', 0, (1., 2., 3.))
    run.abort_episode('Startup timeout', coverage=.01).result(5)
    run.close()
    root = tmp_path / 'reports'
    names = ['20261009T080419Z_first', '20261009T113811Z_second', 'recovered_failure']
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        shutil.copyfile(run.report.directory / 'episodes.csv', directory / 'episodes.csv')
    (root / names[-1] / 'source_run.json').write_text(json.dumps({
        'original_run_timestamp': '20261009T080419Z'}))
    ordered = [root / name / 'episodes.csv' for name in (names[0], names[2], names[1])]
    assert discover_reports(root) == ordered
    before = [path.read_bytes() for path in ordered]
    output = root / 'all_runs'
    args = ['--all-runs', str(root), '--output-dir', str(output)]
    main(args)
    main(args)
    assert discover_reports(root) == ordered
    rows = read_records(output / 'episodes.csv')
    assert len(rows) == 3 and not any(row['success'] for row in rows)
    assert [row['episode'] for row in rows] == [1, 2, 3]
    assert [path.read_bytes() for path in ordered] == before
    assert len(list(output.glob('*.png'))) == len(list(output.glob('*.pdf'))) == 4


def test_all_runs_empty_directory_is_rejected(tmp_path):
    """Do not create an empty success report when no measurements exist."""
    output = tmp_path / 'all_runs'
    with pytest.raises(SystemExit):
        main(['--all-runs', str(tmp_path), '--output-dir', str(output)])
    assert not output.exists()
