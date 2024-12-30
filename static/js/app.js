const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const settingsBtn = document.getElementById('settings-btn');
const settingsPanel = document.getElementById('settings-panel');
const statusDiv = document.getElementById('status');
const videoFeed = document.getElementById('video-feed');
const captureImage = document.getElementById('capture-image');

const showLandmarks = document.getElementById('show-landmarks');
const showEAR = document.getElementById('show-ear');
const showBlinkCount = document.getElementById('show-blink-count');

const earThreshold = document.getElementById('ear-threshold');
const minBlinks = document.getElementById('min-blinks');
const maxBlinks = document.getElementById('max-blinks');

let statusCheckInterval;

settingsBtn.addEventListener('click', () => {
    settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
});

async function updateSettings() {
    try {
        const minValue = parseInt(minBlinks.value);
        const maxValue = parseInt(maxBlinks.value);
        
        if (minValue >= maxValue) {
            alert('Minimum blink count must be less than maximum blink count');
            return;
        }

        await fetch('/update_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                show_landmarks: showLandmarks.checked,
                show_ear: showEAR.checked,
                show_blink_count: showBlinkCount.checked,
                ear_threshold: parseFloat(earThreshold.value),
                min_blinks: minValue,
                max_blinks: maxValue
            })
        });
    } catch (error) {
        console.error('Error updating settings:', error);
    }
}

showLandmarks.addEventListener('change', updateSettings);
showEAR.addEventListener('change', updateSettings);
showBlinkCount.addEventListener('change', updateSettings);
earThreshold.addEventListener('change', updateSettings);
minBlinks.addEventListener('change', updateSettings);
maxBlinks.addEventListener('change', updateSettings);

startBtn.addEventListener('click', async () => {
    try {
        captureImage.style.display = 'none';
        statusDiv.style.display = 'none';
        
        const response = await fetch('/start');
        const data = await response.json();
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        videoFeed.src = '/video_feed';
        videoFeed.style.display = 'block';
        
        statusCheckInterval = setInterval(checkStatus, 1000);
    } catch (error) {
        console.error('Error starting session:', error);
        statusDiv.textContent = 'Error starting camera';
    }
});

stopBtn.addEventListener('click', async () => {
    try {
        await fetch('/stop');
        stopSession();
    } catch (error) {
        console.error('Error stopping session:', error);
    }
});

async function checkStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        if (data.completed) {
            const captureResponse = await fetch('/capture');
            const captureData = await captureResponse.json();
            
            if (captureData.image) {
                captureImage.src = `data:image/jpeg;base64,${captureData.image}`;
                videoFeed.style.display = 'none';
                captureImage.style.display = 'block';
                
                statusDiv.textContent = `Blinks: ${data.target}`;
                statusDiv.style.display = 'block';
            }
            
            stopSession();
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

function stopSession() {
    clearInterval(statusCheckInterval);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    videoFeed.style.display = 'none';
}