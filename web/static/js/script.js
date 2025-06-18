document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const emotionSpan = document.getElementById('emotion');
    const loading = document.getElementById('loading');
    let emotionChart = null;

    analyzeBtn.addEventListener('click', function() {
        const text = textInput.value.trim();
        
        if (text === '') {
            alert('Please enter some text to analyze');
            return;
        }

        // Show loading
        loading.style.display = 'block';
        resultSection.style.display = 'none';

        // Send request to backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            loading.style.display = 'none';
            
            // Show results
            resultSection.style.display = 'block';
            
            // Update emotion text
            emotionSpan.textContent = data.emotion;
            
            // Create emotion color mapping
            const emotionColors = {
                'anger': 'rgba(231, 76, 60, 0.7)',
                'disgust': 'rgba(142, 68, 173, 0.7)',
                'fear': 'rgba(52, 73, 94, 0.7)',
                'joy': 'rgba(46, 204, 113, 0.7)',
                'neutral': 'rgba(149, 165, 166, 0.7)',
                'sadness': 'rgba(41, 128, 185, 0.7)',
                'shame': 'rgba(243, 156, 18, 0.7)',
                'surprise': 'rgba(230, 126, 34, 0.7)'
            };
            
            // Prepare chart data
            const labels = Object.keys(data.probabilities);
            const values = Object.values(data.probabilities);
            const backgroundColors = labels.map(label => emotionColors[label] || 'rgba(0, 0, 0, 0.7)');
            
            // Create or update chart
            const ctx = document.getElementById('emotion-chart').getContext('2d');
            
            if (emotionChart) {
                emotionChart.destroy();
            }
            
            emotionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Emotion Probability',
                        data: values,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            loading.style.display = 'none';
            alert('An error occurred while analyzing the text. Please try again.');
        });
    });
});