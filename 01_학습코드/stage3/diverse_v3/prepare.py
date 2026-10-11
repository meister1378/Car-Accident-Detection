"""KITTI synced RGB/OXTS -> time-aligned weak-label manifest. No network access."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_times(path):
    values = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    times = np.array(values, dtype='datetime64[ns]').astype(np.int64) / 1e9
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError(f'Invalid timestamps: {path}')
    return times


def labels(speed, yaw_left, cfg):
    """Yaw convention: radians/sec, positive LEFT. Thresholds are hypotheses."""
    speed = np.asarray(speed, dtype=float)
    yaw_left = np.asarray(yaw_left, dtype=float)
    if not (np.isfinite(speed).all() and np.isfinite(yaw_left).all()):
        raise ValueError('Nonfinite telemetry')
    w = cfg['speed_smoothing_frames']
    if w < 1 or w % 2 != 1:
        raise ValueError('Smoothing window must be positive and odd')
    smooth = np.convolve(np.pad(speed, (w//2, w//2), mode='edge'), np.ones(w)/w, mode='valid')
    acceleration = np.gradient(smooth, 1 / cfg['fps'])
    accel = np.full(len(speed), 2, dtype=np.int64)
    accel[acceleration > cfg['accel_threshold_mps2']] = 0
    accel[acceleration < -cfg['accel_threshold_mps2']] = 1
    stopped = False
    for i, v in enumerate(speed):
        if v < cfg['stop_enter_mps']:
            stopped = True
        elif v >= cfg['stop_exit_mps']:
            stopped = False
        if stopped:
            accel[i] = 3
    threshold = np.deg2rad(cfg['yaw_threshold_degps'])
    steer = np.full(len(speed), 1, dtype=np.int64)
    steer[yaw_left > threshold] = 0
    steer[yaw_left < -threshold] = 2
    return accel, steer, acceleration


def prepare(root, output, cfg, val_dates):
    records = []
    for image_dir in sorted(Path(root).rglob('image_02/data')):
        drive = image_dir.parent.parent
        oxts_dir = drive / 'oxts/data'
        if not oxts_dir.exists() or not drive.name.endswith('_sync'):
            continue
        image_times = read_times(image_dir.parent / 'timestamps.txt')
        sensor_times = read_times(oxts_dir.parent / 'timestamps.txt')
        images = sorted(image_dir.glob('*.png'))
        sensor_files = sorted(oxts_dir.glob('*.txt'))
        if len(images) != len(image_times) or len(sensor_files) != len(sensor_times):
            raise ValueError(f'File/timestamp count mismatch: {drive}')
        # Refuse silent alignment shifts due to missing files.
        if [int(p.stem) for p in images] != list(range(len(images))):
            raise ValueError(f'Noncontiguous image indices: {drive}')
        if [int(p.stem) for p in sensor_files] != list(range(len(sensor_files))):
            raise ValueError(f'Noncontiguous OXTS indices: {drive}')
        sensor = np.stack([np.loadtxt(p) for p in sensor_files])
        if sensor.shape[1] != 30 or not np.isfinite(sensor).all():
            raise ValueError(f'Expected 30 finite OXTS fields: {drive}')
        start, end = max(image_times[0], sensor_times[0]), min(image_times[-1], sensor_times[-1])
        grid = start + np.arange(int(np.floor((end-start)*cfg['fps']))+1) / cfg['fps']
        if len(grid) < cfg['clip_frames']:
            continue
        right = np.clip(np.searchsorted(image_times, grid), 1, len(image_times)-1)
        indices = np.where(abs(image_times[right]-grid) < abs(image_times[right-1]-grid), right, right-1)
        sr = np.clip(np.searchsorted(sensor_times, grid), 1, len(sensor_times)-1)
        valid = ((sensor_times[sr]-sensor_times[sr-1]) <= cfg['max_sensor_gap_seconds'])
        valid &= abs(image_times[indices]-grid) <= 0.06
        # Keep a contiguous time series; reject recordings with gaps instead of bridging them.
        if not valid.all():
            print('SKIP timestamps/gaps:', drive.name)
            continue
        # KITTI OXTS: vf = field 9, wz = field 20 (one-based); x forward/y left/z up.
        speed = np.interp(grid, sensor_times, sensor[:, 8])
        if np.any(speed < -0.1):
            print('SKIP reverse driving:', drive.name)
            continue
        yaw = np.interp(grid, sensor_times, sensor[:, 19])
        accel, steer, derivative = labels(speed, yaw, cfg)
        date = drive.name[:10]
        records.append(dict(id=drive.name, group=date, split='val' if date in val_dates else 'train',
                            images=[str(images[i].resolve()) for i in indices],
                            seconds=(grid-grid[0]).tolist(), accel=accel.tolist(), steer=steer.tolist(),
                            speed_mps=speed.tolist(), acceleration_mps2=derivative.tolist(),
                            yaw_left_radps=yaw.tolist(), label_source='KITTI_OXTS_weak_thresholds'))
    if not records:
        raise ValueError('No usable synced KITTI recordings. Expected date/drive_sync/image_02/data and oxts/data.')
    splits = {r['split'] for r in records}
    if splits != {'train', 'val'}:
        raise ValueError('Need at least one train date and one validation date; change VAL_DATES or add recordings.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'config': cfg, 'records': records}, ensure_ascii=False), encoding='utf-8')
    report = {}
    for split in ['train', 'val']:
        subset = [r for r in records if r['split'] == split]
        a = np.concatenate([r['accel'] for r in subset])
        s = np.concatenate([r['steer'] for r in subset])[a != 3]
        report[split] = dict(recordings=len(subset), frames=len(a),
                             accel_counts=dict(zip(cfg['accel_classes'], np.bincount(a, minlength=4).tolist())),
                             moving_steer_counts=dict(zip(cfg['steer_classes'], np.bincount(s, minlength=3).tolist())))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='config.json')
    p.add_argument('--val-dates', nargs='+', required=True)
    a = p.parse_args()
    prepare(a.root, a.output, json.loads(Path(a.config).read_text()), set(a.val_dates))
