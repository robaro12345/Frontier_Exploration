"""Export measured episode results and report-ready training figures."""

import argparse
import csv
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = ('episode', 'cumulative_reward', 'steps', 'coverage', 'success', 'epsilon',
           'navigation_successes', 'optimizer_updates', 'outcome', 'reason',
           'world', 'pose_index', 'spawn_pose')


def summarize(records, final_epsilon):
    """Include failed attempts; leave undefined means unavailable for empty runs."""
    count = len(records)
    return {
        'total_episodes': count,
        'completed_episodes': sum(row['success'] for row in records),
        'mean_episode_reward': (math.fsum(row['cumulative_reward'] for row in records)
                                / count if count else None),
        'success_rate': sum(row['success'] for row in records) / count if count else None,
        'mean_steps_per_episode': sum(row['steps'] for row in records) / count if count else None,
        'final_epsilon': final_epsilon,
    }


def _number(value, percent=False):
    if value is None:
        return 'N/A'
    return f'{100 * value:.2f}%' if percent else f'{value:.6f}'


def summary_text(summary):
    """Use the requested labels, with explicit percentage units."""
    return ('Training Summary\n'
            f'Total Episodes: {summary["total_episodes"]}\n'
            f'Mean Episode Reward: {_number(summary["mean_episode_reward"])}\n'
            f'Success Rate: {_number(summary["success_rate"], percent=True)}\n'
            f'Mean Steps per Episode: {_number(summary["mean_steps_per_episode"])}\n'
            f'Final Epsilon: {_number(summary["final_epsilon"])}')


def episode_text(row):
    """Format one actual attempt without substituting missing coverage with zero."""
    context = (f'World: {row["world"]}\nPose Index: {row["pose_index"]}\n'
               if row['world'] else '')
    return (f'Episode: {row["episode"]}\n'
            f'Cumulative Reward: {_number(row["cumulative_reward"])}\n'
            f'Steps: {row["steps"]}\n'
            f'Coverage: {_number(row["coverage"], percent=True)}\n'
            f'Success: {row["success"]}\n'
            f'Epsilon: {_number(row["epsilon"])}\n'
            f'Navigation Successes: {row["navigation_successes"]}\n'
            f'Outcome: {row["outcome"]}\nReason: {row["reason"]}\n' + context).rstrip()


def _atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def _write_csv(directory, records):
    temporary = directory / 'episodes.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(directory / 'episodes.csv')


def _write_summary(directory, records, final_epsilon, scope):
    summary = summarize(records, final_epsilon)
    _atomic_text(directory / 'summary.json',
                 json.dumps(summary, indent=2, allow_nan=False) + '\n')
    episodes = '\n\n'.join(episode_text(row) for row in records)
    definitions = (
        'Coverage = known cells / all cells in the latest received occupancy grid.\n'
        'Success = detector completion without an abort; not proof of full world coverage.\n'
        'Aborted/interrupted attempts count as failures; unfinished actions are not steps.\n'
        'Success-rate graph = cumulative successful episodes / recorded attempts.\n'
        + scope + '\n')
    _atomic_text(directory / 'training_report.txt',
                 definitions + '\n' + (episodes + '\n\n' if episodes else '')
                 + summary_text(summary) + '\n')
    return summary


class EpisodeReport:
    """Keep each process's measurements separate, including checkpoint resumes."""

    def __init__(self, checkpoint_directory):
        """Create a unique run directory while preserving previous report files."""
        root = Path(checkpoint_directory) / 'reports'
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = Path(tempfile.mkdtemp(prefix=timestamp + '_', dir=root))
        self.records = []

    def append(self, **values):
        """Persist full-precision measurements before the next world starts."""
        row = dict(episode=len(self.records) + 1, **values)
        self.records.append(row)
        _write_csv(self.directory, self.records)
        self.write_summary(row['epsilon'])
        return row

    def write_summary(self, final_epsilon):
        """Update readable and machine-readable summaries from the same records."""
        return _write_summary(
            self.directory, self.records, final_epsilon,
            'These results cover this process only; resumed runs have separate reports.')


def read_records(csv_path):
    """Read the logger's CSV to regenerate figures without ROS or PyTorch."""
    with Path(csv_path).open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key in ('episode', 'steps', 'navigation_successes', 'optimizer_updates'):
            row[key] = int(row[key])
        for key in ('cumulative_reward', 'epsilon'):
            row[key] = float(row[key])
        row['coverage'] = float(row['coverage']) if row['coverage'] else None
        if row['success'] not in ('True', 'False'):
            raise ValueError('CSV success must be True or False')
        row['success'] = row['success'] == 'True'
    return records


def plot_records(records, directory):
    """Save raw reward/step curves and measured cumulative episode success rate."""
    if not records:
        return []
    import matplotlib
    matplotlib.use('Agg')  # No GUI or ROS callback is required for report export.
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episodes = [row['episode'] for row in records]
    successes = 0
    rates = []
    for index, row in enumerate(records, start=1):
        successes += int(row['success'])
        rates.append(100 * successes / index)
    series = [
        ('reward_vs_episode', 'Reward vs. Episode', 'Cumulative reward per episode',
         [row['cumulative_reward'] for row in records]),
        ('success_rate_vs_episode', 'Success Rate vs. Episode',
         'Cumulative episode success rate (%)', rates),
        ('steps_per_episode', 'Steps/Episode vs. Episode', 'Completed frontier actions',
         [row['steps'] for row in records]),
    ]
    paths = []
    combined, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    try:
        for axis, (name, title, label, values) in zip(axes, series):
            figure, single = plt.subplots(figsize=(8, 4), constrained_layout=True)
            try:
                for target in (single, axis):
                    target.plot(episodes, values, marker='o', markersize=4, linewidth=1.5)
                    target.set(title=title, xlabel='Episode', ylabel=label)
                    target.xaxis.set_major_locator(MaxNLocator(integer=True))
                    target.set_xlim(min(episodes) - .5, max(episodes) + .5)
                    target.grid(True, alpha=.25)
                    if name == 'success_rate_vs_episode':
                        target.set_ylim(-2, 102)
                    elif name == 'steps_per_episode':
                        target.set_ylim(0, max(values) + max(1, max(values) * .1))
                        target.yaxis.set_major_locator(MaxNLocator(integer=True))
                for suffix in ('png', 'pdf'):
                    path = directory / f'{name}.{suffix}'
                    figure.savefig(path, dpi=200)
                    paths.append(path)
            finally:
                plt.close(figure)
        for suffix in ('png', 'pdf'):
            path = directory / f'training_curves.{suffix}'
            combined.savefig(path, dpi=200)
            paths.append(path)
    finally:
        plt.close(combined)
    return paths


def combine_reports(csv_paths, directory):
    """Combine explicitly ordered run histories, retaining every failed attempt."""
    sources = [Path(path).resolve() for path in csv_paths]
    directory = Path(directory).resolve()
    if len(sources) != len(set(sources)):
        raise ValueError('The same report cannot be included twice')
    if directory in {source.parent for source in sources}:
        raise ValueError('Combined output must be separate from the source reports')
    records = []
    for source in sources:
        for row in read_records(source):
            records.append(dict(row, episode=len(records) + 1))
    if not records:
        raise ValueError('No recorded attempts to combine')
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory, records)
    scope = ('Combined attempt history; episode numbers include retries and failures.\n'
             'Sources in chronological order:\n' + '\n'.join(str(p) for p in sources))
    summary = _write_summary(directory, records, records[-1]['epsilon'], scope)
    _atomic_text(directory / 'sources.json',
                 json.dumps([str(path) for path in sources], indent=2) + '\n')
    return summary, plot_records(records, directory)


def discover_reports(root):
    """Find original run CSVs chronologically, excluding previous merged outputs."""
    sources = []
    for source in Path(root).glob('*/episodes.csv'):
        directory = source.parent
        if (directory / 'sources.json').exists():
            continue
        metadata = directory / 'source_run.json'
        if metadata.exists():
            timestamp = json.loads(metadata.read_text())['original_run_timestamp']
            recovered = 1
        else:
            timestamp = directory.name[:16]
            recovered = 0
        datetime.strptime(timestamp, '%Y%m%dT%H%M%SZ')
        sources.append(((timestamp, recovered, directory.name), source))
    if not sources:
        raise ValueError(f'No original run CSVs found in {root}')
    return [source for _, source in sorted(sources)]


def main(args=None):
    """Export one run's graphs or combine chronological run CSVs after training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='*',
                        help='Recorded episodes.csv files, oldest run first')
    parser.add_argument('--all-runs', type=Path,
                        help='Merge all original run CSVs in a reports directory')
    parser.add_argument('--output-dir', type=Path, help='Default: beside the CSV')
    options = parser.parse_args(args)
    if options.all_runs is not None:
        if options.csv:
            parser.error('Use --all-runs or explicit CSV paths, not both')
        try:
            options.csv = discover_reports(options.all_runs)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
    elif not options.csv:
        parser.error('Provide a CSV path or --all-runs')
    if options.all_runs is not None or len(options.csv) > 1:
        if options.output_dir is None:
            parser.error('--output-dir is required when combining runs')
        try:
            summary, paths = combine_reports(options.csv, options.output_dir)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(summary_text(summary))
        print((options.output_dir / 'training_report.txt').resolve())
        print((options.output_dir / 'episodes.csv').resolve())
        for path in paths:
            print(path.resolve())
        return
    records = read_records(options.csv[0])
    if not records:
        parser.error('No recorded episodes; graphs require actual measurements')
    for path in plot_records(records, options.output_dir or options.csv[0].parent):
        print(path.resolve())


if __name__ == '__main__':
    main()
