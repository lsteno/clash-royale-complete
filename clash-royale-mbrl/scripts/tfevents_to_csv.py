#!/usr/bin/env python3
"""Convert TensorBoard events file to CSV format."""

import argparse
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def convert_tfevents_to_csv(events_path: str, output_path: str = None):
    """Convert a tfevents file to CSV."""
    events_path = Path(events_path)
    if output_path is None:
        output_path = events_path.with_suffix('.csv')
    else:
        output_path = Path(output_path)
    
    print(f"Loading events from: {events_path}")
    
    # Load the events file
    ea = EventAccumulator(str(events_path))
    ea.Reload()
    
    # Get all scalar tags
    tags = ea.Tags().get('scalars', [])
    print(f"Found {len(tags)} scalar tags: {tags}")
    
    if not tags:
        print("No scalar data found in events file.")
        return
    
    # Collect all data points
    rows = []
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            rows.append({
                'wall_time': event.wall_time,
                'step': event.step,
                'tag': tag,
                'value': event.value
            })
    
    # Sort by step then tag
    rows.sort(key=lambda x: (x['step'], x['tag']))
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['wall_time', 'step', 'tag', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to: {output_path}")
    
    # Also create a pivoted version (one row per step, columns for each tag)
    pivot_path = output_path.with_stem(output_path.stem + '_pivoted')
    
    # Group by step
    from collections import defaultdict
    step_data = defaultdict(dict)
    for row in rows:
        step_data[row['step']][row['tag']] = row['value']
        step_data[row['step']]['_wall_time'] = row['wall_time']
    
    # Write pivoted CSV
    all_tags = sorted(tags)
    with open(pivot_path, 'w', newline='') as f:
        fieldnames = ['step', 'wall_time'] + all_tags
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(step_data.keys()):
            row = {'step': step, 'wall_time': step_data[step].get('_wall_time', '')}
            for tag in all_tags:
                row[tag] = step_data[step].get(tag, '')
            writer.writerow(row)
    
    print(f"Wrote pivoted version to: {pivot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorBoard events to CSV")
    parser.add_argument("events_file", help="Path to tfevents file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: same name with .csv)")
    args = parser.parse_args()
    
    convert_tfevents_to_csv(args.events_file, args.output)
