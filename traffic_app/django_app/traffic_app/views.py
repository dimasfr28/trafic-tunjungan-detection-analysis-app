import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.conf import settings
import pandas as pd


def get_csv_data():
    """Load CSV data"""
    csv_path = settings.CSV_PATH
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['datetime_start'] = pd.to_datetime(df['datetime_start'])
        df['hour'] = df['datetime_start'].dt.hour
        df['minute'] = df['datetime_start'].dt.minute
        df['day_name'] = df['datetime_start'].dt.day_name()
        df['day_of_week'] = df['datetime_start'].dt.dayofweek
        df['total'] = df['count_of_car'] + df['count_of_motorcycle'] + df['count_of_heavy']
        return df
    return None


def get_nearest_time_bucket(current_time):
    """Get nearest 10-minute bucket"""
    minute = current_time.minute
    hour = current_time.hour

    if minute < 30:
        # Round down to previous bucket
        bucket_minute = (minute // 10) * 10
        if bucket_minute == 0:
            bucket_minute = 0
    else:
        # Round up to next bucket
        bucket_minute = ((minute // 10) + 1) * 10
        if bucket_minute >= 60:
            bucket_minute = 0
            hour = (hour + 1) % 24

    return hour, bucket_minute


def get_video_files():
    """Get all video files from directory sorted by datetime"""
    video_dir = settings.VIDEO_DIR
    if not video_dir.exists():
        return []

    videos = []
    for f in video_dir.glob('*.mp4'):
        # Parse datetime from filename: screen_recording_20251105_120456_seg2.mp4
        # Format: YYYYMMDD_HHMMSS
        match = re.search(r'(\d{8})_(\d{6})', f.name)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS

            # Parse full datetime
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])

            try:
                dt = datetime(year, month, day, hour, minute, second)
                videos.append({
                    'path': str(f),
                    'name': f.name,
                    'datetime': dt,
                    'hour': hour,
                    'minute': minute
                })
            except ValueError:
                # Skip invalid dates
                continue

    # Sort by datetime (newest first)
    return sorted(videos, key=lambda x: x['datetime'], reverse=True)


def get_nearest_video(current_hour, current_minute):
    """
    Get ALL videos with hour closest to current time (from all dates).
    Strategy:
    1. Calculate time difference based on hour:minute only (ignore date)
    2. Find the closest hour range
    3. Return ALL videos in that hour range, sorted by filename
    """
    videos = get_video_files()
    if not videos:
        return None

    # Calculate time difference for ALL videos based on hour:minute only (ignore date)
    target_time_minutes = current_hour * 60 + current_minute

    for v in videos:
        video_time_minutes = v['hour'] * 60 + v['minute']
        # Calculate absolute difference in minutes
        time_diff_minutes = abs(video_time_minutes - target_time_minutes)
        v['time_diff_minutes'] = time_diff_minutes

    # Sort by time difference to find the closest hour
    videos_sorted = sorted(videos, key=lambda x: x['time_diff_minutes'])

    if not videos_sorted:
        return None

    # Get the closest time difference (in minutes)
    closest_diff = videos_sorted[0]['time_diff_minutes']

    # Get ALL videos within the same hour range (within 30 minutes of the closest)
    threshold_minutes = 30

    matching_videos = [v for v in videos if v['time_diff_minutes'] <= closest_diff + threshold_minutes]

    # Sort by filename (this will group by date and time naturally)
    matching_videos.sort(key=lambda x: x['name'])

    return matching_videos


def home(request):
    """Home page view"""
    return render(request, 'home.html')


def analysis(request):
    """Traffic Analysis page view"""
    return render(request, 'analysis.html')


def prediction(request):
    """Prediction page view"""
    return render(request, 'prediction.html')


def get_video_for_hour(request, hour):
    """API to get video file for specific hour"""
    try:
        hour = int(hour)
        videos = get_video_files()
        hour_videos = [v for v in videos if v['hour'] == hour]

        if hour_videos:
            return JsonResponse({
                'success': True,
                'videos': [v['name'] for v in hour_videos]
            })
        else:
            # Return any available videos
            return JsonResponse({
                'success': True,
                'videos': [v['name'] for v in videos[:6]]
            })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def get_metrics(request):
    """API to get traffic metrics"""
    try:
        df = get_csv_data()
        if df is None:
            return JsonResponse({'success': False, 'error': 'CSV not found'})

        now = datetime.now()
        current_hour, current_bucket = get_nearest_time_bucket(now)
        day_name = now.strftime('%A')
        day_of_week = now.weekday()

        # 1. Average Vehicle Count per Nearest Time Period
        bucket_data = df[(df['hour'] == current_hour) & (df['minute'] <= 10)]
        if len(bucket_data) > 0:
            avg_vehicle = bucket_data['total'].mean()
        else:
            avg_vehicle = df['total'].mean()

        # 2. Peak Hour Range
        hourly_clusters = df.groupby('hour')['cluster'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 1)
        peak_hours = hourly_clusters[hourly_clusters == 3].index.tolist()

        if peak_hours:
            # Group consecutive hours
            peak_ranges = []
            start = peak_hours[0]
            end = peak_hours[0]
            for h in peak_hours[1:]:
                if h == end + 1:
                    end = h
                else:
                    peak_ranges.append(f"{start:02d}:00-{end+1:02d}:00")
                    start = h
                    end = h
            peak_ranges.append(f"{start:02d}:00-{end+1:02d}:00")
            peak_hour_range = ", ".join(peak_ranges)
        else:
            # Get highest cluster hours
            high_hours = hourly_clusters[hourly_clusters == hourly_clusters.max()].index.tolist()
            if high_hours:
                peak_hour_range = f"{min(high_hours):02d}:00-{max(high_hours)+1:02d}:00"
            else:
                peak_hour_range = "N/A"

        # 3. Peak Day
        daily_clusters = df.groupby('day_name')['cluster'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 1)
        cluster_3_days = daily_clusters[daily_clusters == 3]

        if len(cluster_3_days) > 0:
            # Count frequency of cluster 3
            day_cluster_counts = df[df['cluster'] == 3].groupby('day_name').size()
            peak_day = day_cluster_counts.idxmax() if len(day_cluster_counts) > 0 else daily_clusters.idxmax()
        else:
            peak_day = daily_clusters.idxmax()

        # 4. Dominant Vehicle Type for Nearest Time Period
        if len(bucket_data) > 0:
            car_total = bucket_data['count_of_car'].sum()
            moto_total = bucket_data['count_of_motorcycle'].sum()
            heavy_total = bucket_data['count_of_heavy'].sum()
        else:
            car_total = df['count_of_car'].sum()
            moto_total = df['count_of_motorcycle'].sum()
            heavy_total = df['count_of_heavy'].sum()

        total_vehicles = car_total + moto_total + heavy_total

        if moto_total >= car_total and moto_total >= heavy_total:
            dominant_type = "Motorcycle"
            dominant_pct = (moto_total / total_vehicles * 100) if total_vehicles > 0 else 0
        elif car_total >= heavy_total:
            dominant_type = "Car"
            dominant_pct = (car_total / total_vehicles * 100) if total_vehicles > 0 else 0
        else:
            dominant_type = "Heavy Vehicle"
            dominant_pct = (heavy_total / total_vehicles * 100) if total_vehicles > 0 else 0

        return JsonResponse({
            'success': True,
            'metrics': {
                'avg_vehicle_count': round(avg_vehicle, 1),
                'current_time_bucket': f"{current_hour:02d}:{current_bucket:02d}",
                'peak_hour_range': peak_hour_range,
                'peak_day': peak_day,
                'dominant_vehicle_type': dominant_type,
                'dominant_vehicle_pct': round(dominant_pct, 1),
                'current_day': day_name
            }
        })
    except Exception as e:
        import traceback
        return JsonResponse({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


def get_chart_data(request):
    """API to get chart data for analysis page"""
    try:
        df = get_csv_data()
        if df is None:
            return JsonResponse({'success': False, 'error': 'CSV not found'})

        selected_day = request.GET.get('day', datetime.now().strftime('%A'))

        # Filter by day
        day_data = df[df['day_name'] == selected_day]

        # 1. Line Chart - Traffic Density per Hour
        hourly_data = day_data.groupby('hour').agg({
            'count_of_car': 'sum',
            'count_of_motorcycle': 'sum',
            'count_of_heavy': 'sum',
            'cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else 1
        }).reset_index()

        line_chart = {
            'labels': [f"{h:02d}:00" for h in range(24)],
            'datasets': {
                'car': [0] * 24,
                'motorcycle': [0] * 24,
                'heavy': [0] * 24,
                'cluster': [1] * 24
            }
        }

        for _, row in hourly_data.iterrows():
            h = int(row['hour'])
            line_chart['datasets']['car'][h] = int(row['count_of_car'])
            line_chart['datasets']['motorcycle'][h] = int(row['count_of_motorcycle'])
            line_chart['datasets']['heavy'][h] = int(row['count_of_heavy'])
            line_chart['datasets']['cluster'][h] = int(row['cluster'])

        # 2. Pie Chart - Cluster Distribution
        cluster_counts = day_data['cluster'].value_counts().to_dict()
        pie_chart = {
            'labels': ['Low (1)', 'Medium (2)', 'High (3)'],
            'data': [
                cluster_counts.get(1, 0),
                cluster_counts.get(2, 0),
                cluster_counts.get(3, 0)
            ]
        }

        # 3. Bar Chart - Total Vehicles per Day
        daily_totals = df.groupby('day_name').agg({
            'count_of_car': 'sum',
            'count_of_motorcycle': 'sum',
            'count_of_heavy': 'sum',
            'cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else 1
        }).reset_index()

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        bar_chart = {
            'labels': day_order,
            'datasets': {
                'car': [],
                'motorcycle': [],
                'heavy': [],
                'cluster': []
            }
        }

        for day in day_order:
            day_row = daily_totals[daily_totals['day_name'] == day]
            if len(day_row) > 0:
                bar_chart['datasets']['car'].append(int(day_row['count_of_car'].iloc[0]))
                bar_chart['datasets']['motorcycle'].append(int(day_row['count_of_motorcycle'].iloc[0]))
                bar_chart['datasets']['heavy'].append(int(day_row['count_of_heavy'].iloc[0]))
                bar_chart['datasets']['cluster'].append(int(day_row['cluster'].iloc[0]))
            else:
                bar_chart['datasets']['car'].append(0)
                bar_chart['datasets']['motorcycle'].append(0)
                bar_chart['datasets']['heavy'].append(0)
                bar_chart['datasets']['cluster'].append(1)

        # 4. Heatmap - Traffic Intensity (Day x Hour)
        heatmap_data = []
        for day_idx, day in enumerate(day_order):
            day_df = df[df['day_name'] == day]
            for hour in range(24):
                hour_df = day_df[day_df['hour'] == hour]
                if len(hour_df) > 0:
                    intensity = hour_df['total'].sum()
                    cluster = hour_df['cluster'].mode()[0] if len(hour_df['cluster'].mode()) > 0 else 1
                else:
                    intensity = 0
                    cluster = 1
                heatmap_data.append({
                    'x': hour,
                    'y': day_idx,
                    'value': int(intensity),
                    'cluster': int(cluster)
                })

        heatmap = {
            'xLabels': [f"{h:02d}" for h in range(24)],
            'yLabels': day_order,
            'data': heatmap_data
        }

        # Get available days
        available_days = df['day_name'].unique().tolist()

        return JsonResponse({
            'success': True,
            'selected_day': selected_day,
            'available_days': available_days,
            'line_chart': line_chart,
            'pie_chart': pie_chart,
            'bar_chart': bar_chart,
            'heatmap': heatmap
        })
    except Exception as e:
        import traceback
        return JsonResponse({'success': False, 'error': str(e), 'trace': traceback.format_exc()})
