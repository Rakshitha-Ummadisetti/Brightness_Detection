const express = require('express');
const bodyParser = require('body-parser');
const Jimp = require('jimp');
const LogisticRegression = require('ml-logistic-regression');
const { Matrix } = require('ml-matrix');
const fs = require('fs');
const path = require('path');

// Load the trained model
const modelPath = 'C:/Users/Raksh/OneDrive/Desktop/Data_Eizen/Bright/Bright_Eizen/model2.json';
const model = loadModel(modelPath);

// Initialize Express app
const app = express();
app.use(bodyParser.json({ limit: '50mb' }));

// Serve the index.html file when accessing the root URL
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Load the trained model
function loadModel(modelPath) {
    const modelData = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    return LogisticRegression.load(modelData);
}

// Simplified HOG implementation
function computeHOG(image) {
    const cellSize = 8;
    const blockSize = 2;
    const numBins = 9;
    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const gradients = [];
    const orientations = [];

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const dx = Jimp.intToRGBA(image.getPixelColor(x + 1, y)).r - Jimp.intToRGBA(image.getPixelColor(x - 1, y)).r;
            const dy = Jimp.intToRGBA(image.getPixelColor(x, y + 1)).r - Jimp.intToRGBA(image.getPixelColor(x, y - 1)).r;
            const gradient = Math.sqrt(dx * dx + dy * dy);
            const orientation = Math.atan2(dy, dx) * (180 / Math.PI) + 180;
            gradients.push(gradient);
            orientations.push(orientation);
        }
    }

    const cellHistograms = [];
    for (let y = 0; y < height - cellSize; y += cellSize) {
        for (let x = 0; x < width - cellSize; x += cellSize) {
            const histogram = new Array(numBins).fill(0);
            for (let cy = 0; cy < cellSize; cy++) {
                for (let cx = 0; cx < cellSize; cx++) {
                    const index = (y + cy) * (width - 2) + (x + cx);
                    const gradient = gradients[index];
                    const orientation = orientations[index];
                    const binIndex = Math.floor(orientation / (360 / numBins)) % numBins;
                    histogram[binIndex] += gradient;
                }
            }
            cellHistograms.push(histogram);
        }
    }

    const hogFeatures = [];
    const cellsPerRow = Math.floor((width - cellSize) / cellSize);
    for (let y = 0; y < cellHistograms.length - cellsPerRow * (blockSize - 1); y += cellsPerRow) {
        for (let x = 0; x < cellsPerRow - (blockSize - 1); x++) {
            const block = [];
            for (let by = 0; by < blockSize; by++) {
                for (let bx = 0; bx < blockSize; bx++) {
                    block.push(...cellHistograms[y + by * cellsPerRow + x + bx]);
                }
            }
            const magnitude = Math.sqrt(block.reduce((sum, val) => sum + val * val, 0));
            const normalizedBlock = block.map(val => val / (magnitude + 1e-6));
            hogFeatures.push(...normalizedBlock);
        }
    }

    return hogFeatures;
}

// Smoothing prediction logic
let predictionHistory = [];
const maxHistoryLength = 5;

function smoothPrediction(newPrediction) {
    predictionHistory.push(newPrediction);

    if (predictionHistory.length > maxHistoryLength) {
        predictionHistory.shift(); // Remove the oldest prediction
    }

    // Calculate the most frequent prediction in the history
    const predictionCounts = predictionHistory.reduce((acc, pred) => {
        acc[pred] = (acc[pred] || 0) + 1;
        return acc;
    }, {});

    // Return the prediction with the highest frequency
    return Object.keys(predictionCounts).reduce((a, b) => predictionCounts[a] > predictionCounts[b] ? a : b);
}

// Process incoming image frames
app.post('/process-frame', async (req, res) => {
    try {
        const imageData = req.body.image.replace(/^data:image\/jpeg;base64,/, '');
        const buffer = Buffer.from(imageData, 'base64');
        const image = await Jimp.read(buffer);

        image.resize(128, 128); // Resize to the fixed dimensions used during training
        image.greyscale();

        const hogFeatures = computeHOG(image);
        const featuresMatrix = new Matrix([hogFeatures]);
        const prediction = model.predict(featuresMatrix);

        // Map prediction index to brightness label
        const labels = ['Low', 'Optimal', 'High'];
        const predictedLabel = labels[prediction[0]];

        // Apply smoothing
        const smoothedLabel = smoothPrediction(predictedLabel);

        res.json({ brightness: smoothedLabel });
    } catch (error) {
        console.error("Error processing frame:", error.message);
        res.status(500).json({ error: "Error processing frame" });
    }
});

// Start the server
app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
