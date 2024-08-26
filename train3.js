const fs = require('fs');
const path = require('path');
const Jimp = require('jimp');
const LogisticRegression = require('ml-logistic-regression');
const { Matrix } = require('ml-matrix');

// Specify the paths to your actual image directories
const highBrightnessDir = 'C:/Users/Raksh/OneDrive/Desktop/Data_Eizen/High';
const optimalBrightnessDir = 'C:/Users/Raksh/OneDrive/Desktop/Data_Eizen/Optimal';
const lowBrightnessDir = 'C:/Users/Raksh/OneDrive/Desktop/Data_Eizen/Low';

// Define the fixed image size for all images
const fixedWidth = 128;
const fixedHeight = 128;

// Simplified HOG implementation
function computeHOG(image) {
    const cellSize = 8; // Size of the cell (8x8 pixels)
    const blockSize = 2; // Block size (2x2 cells)
    const numBins = 9; // Number of bins for the histogram
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

async function extractFeatures(imagePath) {
    try {
        console.log(`Processing image: ${imagePath}`);
        const image = await Jimp.read(imagePath);
        // Resize to fixed dimensions to ensure consistency
        image.resize(fixedWidth, fixedHeight);
        image.greyscale();
        const hogFeatures = computeHOG(image);

        // Log the length of the HOG features to check consistency
        console.log(`HOG feature length for ${imagePath}: ${hogFeatures.length}`);

        return hogFeatures;
    } catch (error) {
        console.error(`Error processing image ${imagePath}:`, error.message);
        return null;
    }
}

async function loadImagesFromDirectory(directory, labelIndex, batchSize = 10) {
    if (!fs.existsSync(directory)) {
        console.error(`Directory does not exist: ${directory}`);
        return { dataset: [], labels: [] };
    }

    const files = fs.readdirSync(directory);
    if (files.length === 0) {
        console.warn(`No files found in directory: ${directory}`);
        return { dataset: [], labels: [] };
    }

    const totalBatches = Math.ceil(files.length / batchSize);
    let allDataset = [];
    let allLabels = [];

    for (let i = 0; i < totalBatches; i++) {
        const batchFiles = files.slice(i * batchSize, (i + 1) * batchSize);
        const batchPromises = batchFiles.map(async (file) => {
            const imagePath = path.join(directory, file);
            const features = await extractFeatures(imagePath);
            return features ? { features, label: labelIndex } : null;
        });

        const batchResults = (await Promise.all(batchPromises)).filter(result => result !== null);
        const batchDataset = batchResults.map(result => result.features);
        const batchLabels = batchResults.map(result => result.label);

        allDataset = allDataset.concat(batchDataset);
        allLabels = allLabels.concat(batchLabels);

        console.log(`Processed batch ${i + 1}/${totalBatches} for ${path.basename(directory)}. Valid images: ${batchResults.length}/${batchFiles.length}`);
    }

    return { dataset: allDataset, labels: allLabels };
}

async function prepareDataset() {
    console.log("Preparing dataset...");
    const highBrightnessData = await loadImagesFromDirectory(highBrightnessDir, 2);
    const optimalBrightnessData = await loadImagesFromDirectory(optimalBrightnessDir, 1);
    const lowBrightnessData = await loadImagesFromDirectory(lowBrightnessDir, 0);

    const allDataset = [
        ...highBrightnessData.dataset,
        ...optimalBrightnessData.dataset,
        ...lowBrightnessData.dataset
    ];

    const allLabels = [
        ...highBrightnessData.labels,
        ...optimalBrightnessData.labels,
        ...lowBrightnessData.labels
    ];

    if (allDataset.length === 0 || allLabels.length === 0) {
        throw new Error("No valid images were processed. Check your image directories and file formats.");
    }

    // Check for consistency in feature lengths
    const featureLength = allDataset[0].length;
    if (!allDataset.every(features => features.length === featureLength)) {
        throw new Error("Inconsistent HOG feature lengths detected. Ensure all images are processed consistently.");
    }

    const dataset = new Matrix(allDataset);
    const labels = Matrix.columnVector(allLabels);

    console.log(`Dataset preparation complete. Total images: ${allDataset.length}`);
    return { dataset, labels };
}

async function trainModel() {
    const { dataset, labels } = await prepareDataset();
    console.log("Training model...");
    const model = new LogisticRegression({ numSteps: 1000, learningRate: 5e-3 });
    model.train(dataset, labels);
    console.log("Model training complete.");
    return model;
}

trainModel().then((model) => {
    const modelJson = JSON.stringify(model.toJSON());
    fs.writeFileSync('model2.json', modelJson);
    console.log('Model trained and saved as model1.json');
}).catch((error) => {
    console.error('Error during training:', error);
});
