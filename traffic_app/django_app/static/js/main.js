// Main JavaScript file for Traffic Analysis Application

// Utility Functions
const Utils = {
    formatTime: (hours, minutes) => {
        return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    },

    getCurrentHour: () => {
        const now = new Date();
        return now.getHours();
    },

    getNearestHour: () => {
        const now = new Date();
        const minute = now.getMinutes();
        let hour = now.getHours();

        if (minute >= 30) {
            hour = (hour + 1) % 24;
        }
        return hour;
    },

    formatNumber: (num) => {
        return num.toLocaleString('id-ID');
    },

    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// API Service
const API = {
    baseUrl: '',
    fastApiUrl: 'http://localhost:8001',

    async getMetrics() {
        try {
            const response = await fetch(`${this.baseUrl}/api/metrics/`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching metrics:', error);
            return { success: false, error: error.message };
        }
    },

    async getChartData(day) {
        try {
            const response = await fetch(`${this.baseUrl}/api/chart-data/?day=${day}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching chart data:', error);
            return { success: false, error: error.message };
        }
    },

    async getVideos() {
        try {
            const response = await fetch(`${this.fastApiUrl}/api/videos`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching videos:', error);
            return { success: false, error: error.message };
        }
    },

    getVideoStreamUrl(hour, detection = true) {
        return `${this.fastApiUrl}/api/video/stream/${hour}?detection=${detection}`;
    },

    getCurrentVideoStreamUrl(detection = true) {
        return `${this.fastApiUrl}/api/video/current?detection=${detection}`;
    }
};

// Chart Configuration
const ChartConfig = {
    colors: {
        car: {
            border: 'rgba(54, 162, 235, 1)',
            background: 'rgba(54, 162, 235, 0.2)'
        },
        motorcycle: {
            border: 'rgba(75, 192, 192, 1)',
            background: 'rgba(75, 192, 192, 0.2)'
        },
        heavy: {
            border: 'rgba(255, 99, 132, 1)',
            background: 'rgba(255, 99, 132, 0.2)'
        },
        cluster: {
            low: 'rgba(75, 192, 192, 0.8)',
            medium: 'rgba(255, 206, 86, 0.8)',
            high: 'rgba(255, 99, 132, 0.8)'
        }
    },

    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#94a3b8'
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(148, 163, 184, 0.1)' }
            },
            y: {
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(148, 163, 184, 0.1)' },
                beginAtZero: true
            }
        }
    }
};

// Video Stream Handler
class VideoStreamHandler {
    constructor(imgElement, loadingElement) {
        this.img = imgElement;
        this.loading = loadingElement;
        this.currentUrl = null;
        this.retryCount = 0;
        this.maxRetries = 3;
    }

    async loadStream(url) {
        if (this.loading) {
            this.loading.style.display = 'flex';
        }

        return new Promise((resolve, reject) => {
            this.img.onload = () => {
                if (this.loading) {
                    this.loading.style.display = 'none';
                }
                this.retryCount = 0;
                this.currentUrl = url;
                resolve();
            };

            this.img.onerror = () => {
                if (this.retryCount < this.maxRetries) {
                    this.retryCount++;
                    console.log(`Retry ${this.retryCount}/${this.maxRetries}`);
                    setTimeout(() => {
                        this.img.src = url + '&retry=' + Date.now();
                    }, 1000);
                } else {
                    if (this.loading) {
                        this.loading.style.display = 'none';
                    }
                    reject(new Error('Failed to load video stream'));
                }
            };

            this.img.src = url;
        });
    }

    stop() {
        this.img.src = '';
        this.currentUrl = null;
    }
}

// Notification Handler
const Notification = {
    show(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;

        // Style
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: #fff;
            font-weight: 500;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        const colors = {
            info: '#3b82f6',
            success: '#22c55e',
            warning: '#f59e0b',
            error: '#ef4444'
        };

        notification.style.background = colors[type] || colors.info;

        document.body.appendChild(notification);

        // Auto remove
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    console.log('Traffic Analysis Application initialized');
});

// Export for use in templates
window.TrafficApp = {
    Utils,
    API,
    ChartConfig,
    VideoStreamHandler,
    Notification
};
