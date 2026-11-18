let scannerStream = null;
let scannerModel = null;
let scannerModelPromise = null;
let scannerIndex = null;
let scannerIndexPromise = null;
let scannerInterval = null;
let scannerBusy = false;
let scannerGeneration = 0;
let lastRecognizedItemName = null;
let scannerEmbeddingHistory = [];
let scannerMatchHistory = [];
let scannerTargetResizeObserver = null;
const scannerReferenceCropScales = window.matchMedia &&
    window.matchMedia("(pointer: coarse)").matches
    ? [0.8]
    : [1, 0.8, 0.62];
const scannerFrameCropScales = [0.7, 0.52, 0.36, 0.24];
const scannerExamplesStorageKey = "cssArScannerExamplesV1";
const scannerMaxLearnedExamples = 4;
let scannerPendingMatch = null;
let scannerTrainingStorageError = null;


function normalizeScannerText(value) {
    return String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}


function getScannerItems() {
    return Array.from(document.querySelectorAll("#toolsSection .tool-card, #equipmentSection .tool-card"))
        .map((card) => {
            const image = card.querySelector("img");
            const title = card.querySelector("h3");
            const paragraphs = card.querySelectorAll("p");
            const labels = [
                title ? title.textContent : "",
                image ? image.alt : "",
                card.getAttribute("data-scan-labels") || ""
            ].join(",").split(",").map(normalizeScannerText).filter(Boolean);

            return {
                name: title ? title.textContent.trim() : "",
                image: image ? image.getAttribute("src") : "",
                imageElement: image,
                labels,
                explanation: paragraphs[0] ? paragraphs[0].textContent.replace(/^Explanation:\s*/i, "").trim() : "",
                function: paragraphs[1] ? paragraphs[1].textContent.replace(/^Function:\s*/i, "").trim() : ""
            };
        });
}


function createScannerRegionTensor(source, cropScale = 0.7) {
    const video = source.tagName === "VIDEO";
    const width = video ? source.videoWidth : source.naturalWidth;
    const height = video ? source.videoHeight : source.naturalHeight;
    if (!width || !height) throw new Error("The camera image is not ready to analyze yet.");

    let visibleWidth = width;
    let visibleHeight = height;
    if (video && source.clientWidth && source.clientHeight) {
        const scale = Math.max(source.clientWidth / width, source.clientHeight / height);
        visibleWidth = source.clientWidth / scale;
        visibleHeight = source.clientHeight / scale;
    }
    const size = Math.floor(Math.min(visibleWidth, visibleHeight) * cropScale);
    if (!size) throw new Error("The scan area is too small to analyze.");

    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Could not prepare the camera image for scanning.");
    context.drawImage(
        source,
        Math.floor((width - size) / 2),
        Math.floor((height - size) / 2),
        size,
        size,
        0,
        0,
        canvas.width,
        canvas.height
    );
    return tf.browser.fromPixels(canvas);
}


async function getTensorEmbedding(region, itemName) {
    const embedding = scannerModel.infer(region, true);
    try {
        const values = await embedding.data();
        let magnitude = 0;
        for (let i = 0; i < values.length; i += 1) magnitude += values[i] * values[i];
        magnitude = Math.sqrt(magnitude);
        if (!magnitude) throw new Error(`Could not analyze the scan area for ${itemName}.`);
        return Array.from(values, (value) => value / magnitude);
    } finally {
        embedding.dispose();
    }
}


async function analyzeScannerRegion(source, itemName, includePredictions = false, cropScale = 0.7) {
    const region = createScannerRegionTensor(source, cropScale);
    try {
        const embedding = await getTensorEmbedding(region, itemName);
        const predictions = includePredictions
            ? await scannerModel.classify(region, 10)
            : [];
        return { embedding, predictions };
    } finally {
        region.dispose();
    }
}


async function getImageEmbedding(image, itemName) {
    if (!image.complete) {
        await new Promise((resolve, reject) => {
            image.addEventListener("load", resolve, { once: true });
            image.addEventListener(
                "error",
                () => reject(new Error(`Could not load the catalog image for ${itemName}.`)),
                { once: true }
            );
        });
    }
    if (!image.naturalWidth || !image.naturalHeight) {
        throw new Error(`Could not load the catalog image for ${itemName}.`);
    }
    const scannerImage = new Image();
    scannerImage.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
        scannerImage.addEventListener("load", resolve, { once: true });
        scannerImage.addEventListener(
            "error",
            () => reject(new Error(`Could not load the catalog image for ${itemName} in canvas-safe mode.`)),
            { once: true }
        );
        scannerImage.src = image.currentSrc || image.src;
    });
    const embeddings = [];
    for (const cropScale of scannerReferenceCropScales) {
        embeddings.push((await analyzeScannerRegion(scannerImage, itemName, false, cropScale)).embedding);
    }
    return embeddings;
}


function encodeScannerEmbedding(embedding) {
    const bytes = new Uint8Array(embedding.length);
    for (let index = 0; index < embedding.length; index += 1) {
        const value = Math.max(-1, Math.min(1, embedding[index]));
        bytes[index] = Math.round((value + 1) * 127.5);
    }

    let binary = "";
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
}


function decodeScannerEmbedding(encoded, expectedLength) {
    if (typeof encoded !== "string") {
        throw new Error("A saved scanner example has an invalid format. Clear saved scanner data and try again.");
    }

    let binary;
    try {
        binary = atob(encoded);
    } catch (error) {
        throw new Error("A saved scanner example is unreadable. Clear saved scanner data and try again.");
    }
    if (binary.length !== expectedLength) {
        throw new Error("A saved scanner example is incompatible with this model. Clear saved scanner data and try again.");
    }

    const embedding = Array.from(binary, (character) => (character.charCodeAt(0) / 127.5) - 1);
    const magnitude = Math.sqrt(embedding.reduce((total, value) => total + value * value, 0));
    if (!magnitude) {
        throw new Error("A saved scanner example is empty. Clear saved scanner data and try again.");
    }
    return embedding.map((value) => value / magnitude);
}


function readScannerExampleStore() {
    let stored;
    try {
        stored = window.localStorage.getItem(scannerExamplesStorageKey);
    } catch (error) {
        throw new Error("Could not access saved scanner examples on this device.");
    }
    if (!stored) return {};

    try {
        const examples = JSON.parse(stored);
        if (!examples || typeof examples !== "object" || Array.isArray(examples)) {
            throw new Error("invalid store");
        }
        return examples;
    } catch (error) {
        throw new Error("Saved scanner examples are corrupted. Clear saved scanner data and try again.");
    }
}


function getSavedScannerEmbeddings(catalogItems, embeddingLength) {
    const savedExamples = readScannerExampleStore();
    return catalogItems.reduce((examplesByItem, item) => {
        const examples = savedExamples[item.name] || [];
        if (!Array.isArray(examples) || examples.length > scannerMaxLearnedExamples) {
            throw new Error(`Saved examples for ${item.name} are invalid. Clear saved scanner data and try again.`);
        }
        examplesByItem[item.name] = examples.map((example) => {
            if (!Array.isArray(example) || example.length !== scannerFrameCropScales.length) {
                throw new Error(`A saved example for ${item.name} is incomplete. Clear saved scanner data and try again.`);
            }
            return example.map((embedding) => decodeScannerEmbedding(embedding, embeddingLength));
        });
        return examplesByItem;
    }, {});
}


function rememberScannerExample(item, embeddings) {
    if (!item || embeddings.length !== scannerFrameCropScales.length) {
        throw new Error("Could not save this scan example.");
    }

    const savedExamples = readScannerExampleStore();
    const examples = savedExamples[item.name] || [];
    if (!Array.isArray(examples)) {
        throw new Error(`Saved examples for ${item.name} are invalid. Clear saved scanner data and try again.`);
    }
    examples.push(embeddings.map(encodeScannerEmbedding));
    savedExamples[item.name] = examples.slice(-scannerMaxLearnedExamples);

    try {
        window.localStorage.setItem(scannerExamplesStorageKey, JSON.stringify(savedExamples));
    } catch (error) {
        throw new Error("Could not save this example. Check available device storage and try again.");
    }

    item.embeddings.push(...embeddings);
    const maxEmbeddingCount = scannerReferenceCropScales.length + scannerMaxLearnedExamples;
    if (item.embeddings.length > maxEmbeddingCount) {
        item.embeddings.splice(
            scannerReferenceCropScales.length,
            item.embeddings.length - maxEmbeddingCount
        );
    }
}


function scannerLabelsMatch(item, className) {
    if (!className) return false;
    return item.labels.some((label) =>
        className === label ||
        (` ${className} `).includes(` ${label} `) ||
        (` ${label} `).includes(` ${className} `)
    );
}


function getScannerLabelEvidence(item, predictions, catalog) {
    return predictions.reduce((bestScore, prediction) => {
        if (prediction.probability < 0.08) return bestScore;
        const className = normalizeScannerText(prediction.className);
        const matchingItems = catalog.filter((candidate) => scannerLabelsMatch(candidate, className));
        return matchingItems.length === 1 && matchingItems[0] === item
            ? Math.max(bestScore, prediction.probability)
            : bestScore;
    }, 0);
}


function getConfirmedScannerMatch(history) {
    const votes = history.reduce((counts, candidate) => {
        if (!candidate) return counts;
        let vote = counts.find((entry) => entry.item.name === candidate.item.name);
        if (!vote) {
            vote = { item: candidate.item, count: 0, similarityTotal: 0 };
            counts.push(vote);
        }
        vote.count += 1;
        vote.similarityTotal += candidate.similarity;
        return counts;
    }, []);
    const confirmed = votes.sort((a, b) => b.count - a.count)[0];
    if (!confirmed || confirmed.count < 3) return null;
    return {
        item: confirmed.item,
        similarity: confirmed.similarityTotal / confirmed.count
    };
}


function isScannerCatalogMatch(best, runnerUp) {
    if (!best || best.similarity < (best.labelEvidence >= 0.45 ? 0.41 : 0.48)) return false;
    const margin = runnerUp ? best.score - runnerUp.score : best.score;
    const minimumMargin = best.labelEvidence >= 0.45 ? 0.025 : 0.04;
    return margin >= minimumMargin;
}


function prepareScannerIndex() {
    if (scannerIndex) return Promise.resolve(scannerIndex);
    if (!scannerIndexPromise) {
        scannerIndexPromise = (async () => {
            const catalogItems = getScannerItems();
            if (!catalogItems.length) throw new Error("No tool or equipment items are available to scan.");
            const incompleteItem = catalogItems.find((item) => !item.name || !item.imageElement);
            if (incompleteItem) {
                throw new Error("Every tool and equipment card must have a title and image before scanning.");
            }

            const items = [];
            for (let index = 0; index < catalogItems.length; index += 1) {
                const item = catalogItems[index];
                items.push({
                    ...item,
                    embeddings: await getImageEmbedding(item.imageElement, item.name)
                });
                await new Promise((resolve) => window.requestAnimationFrame(resolve));
            }
            if (items.length !== catalogItems.length) {
                throw new Error("Could not prepare every tool and equipment item for scanning.");
            }
            let learnedExamples = {};
            scannerTrainingStorageError = null;
            try {
                learnedExamples = getSavedScannerEmbeddings(items, items[0].embeddings[0].length);
            } catch (error) {
                scannerTrainingStorageError = error.message || "Saved scanner examples could not be loaded.";
            }
            items.forEach((item) => item.embeddings.push(...(learnedExamples[item.name] || [])));
            scannerIndex = items;
            return items;
        })().catch((error) => {
            scannerIndexPromise = null;
            throw error;
        });
    }
    return scannerIndexPromise;
}


function toggleScannerTraining() {
    if (!scannerStream || !scannerIndex) {
        updateScanResult("Start the camera first", "The scanner needs a live camera view before it can learn an example.");
        return;
    }
    const panel = document.getElementById("scannerTraining");
    const select = document.getElementById("scannerTrainingItem");
    if (!panel.hidden) {
        panel.hidden = true;
        return;
    }

    select.textContent = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose the item shown in the camera";
    select.appendChild(placeholder);
    scannerIndex.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        select.appendChild(option);
    });

    document.getElementById("scannerTrainingStatus").textContent = scannerTrainingStorageError
        ? `${scannerTrainingStorageError} Reset learned examples to enable training.`
        : "Examples are stored on this device. No camera photos are saved.";
    panel.hidden = false;
}


function clearScannerTrainingExamples() {
    const status = document.getElementById("scannerTrainingStatus");
    try {
        window.localStorage.removeItem(scannerExamplesStorageKey);
    } catch (error) {
        status.textContent = "Could not reset saved examples on this device.";
        return;
    }

    scannerIndex = null;
    scannerIndexPromise = null;
    scannerTrainingStorageError = null;
    stopScanner();
    updateScanResult("Learned examples reset", "Start the scanner again to rebuild its catalog index.");
}


async function captureScannerTrainingExample() {
    const status = document.getElementById("scannerTrainingStatus");
    const selectedName = document.getElementById("scannerTrainingItem").value;
    const item = scannerIndex && scannerIndex.find((candidate) => candidate.name === selectedName);
    if (!item) {
        status.textContent = "Choose the item shown in the camera first.";
        return;
    }
    if (!scannerStream) {
        status.textContent = "Start the camera before saving an example.";
        return;
    }
    if (scannerBusy) {
        status.textContent = "Wait for the current camera frame to finish, then try again.";
        return;
    }

    const video = document.getElementById("cameraVideo");
    const button = document.getElementById("scannerTrainingSave");
    const generation = scannerGeneration;
    scannerBusy = true;
    button.disabled = true;
    if (scannerInterval) {
        window.clearInterval(scannerInterval);
        scannerInterval = null;
    }

    try {
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            throw new Error("The camera is not ready yet. Wait for the preview and try again.");
        }

        const embeddings = [];
        for (const cropScale of scannerFrameCropScales) {
            const result = await analyzeScannerRegion(video, "training frame", false, cropScale);
            embeddings.push(result.embedding);
        }
        if (generation !== scannerGeneration || !scannerStream) return;

        rememberScannerExample(item, embeddings);
        scannerEmbeddingHistory = [];
        scannerMatchHistory = [];
        status.textContent = `Saved an example for ${item.name}. Future scans can use up to ${scannerMaxLearnedExamples} camera examples per item.`;
    } catch (error) {
        if (generation !== scannerGeneration) return;
        status.textContent = error.message || "Could not save this camera example.";
    } finally {
        if (generation === scannerGeneration) {
            scannerBusy = false;
            button.disabled = false;
            if (scannerStream && !scannerInterval) {
                scannerInterval = window.setInterval(scanCameraFrame, 1200);
            }
        }
    }
}


function smoothScannerEmbeddings(embeddings) {
    scannerEmbeddingHistory.push(embeddings);
    if (scannerEmbeddingHistory.length > 3) scannerEmbeddingHistory.shift();

    return embeddings.map((embedding, scaleIndex) => {
        const averaged = embedding.map((_, index) =>
            scannerEmbeddingHistory.reduce((total, frame) => total + frame[scaleIndex][index], 0) /
            scannerEmbeddingHistory.length
        );
        const magnitude = Math.sqrt(averaged.reduce((total, value) => total + value * value, 0));
        if (!magnitude) throw new Error("Could not stabilize the camera image.");
        return averaged.map((value) => value / magnitude);
    });
}


function getScannerItemSimilarity(item, frameEmbeddings) {
    const scaleScores = frameEmbeddings.map((frameEmbedding) =>
        Math.max(...item.embeddings.map((itemEmbedding) =>
            itemEmbedding.reduce((score, value, index) => score + value * frameEmbedding[index], 0)
        ))
    );
    scaleScores.sort((a, b) => b - a);
    const bestScaleCount = Math.min(2, scaleScores.length);
    return scaleScores.slice(0, bestScaleCount).reduce((total, score) => total + score, 0) / bestScaleCount;
}


function updateScannerTarget(video) {
    const target = document.querySelector(".scan-target");
    if (!target || !video) return;
    const side = Math.floor(Math.min(video.clientWidth, video.clientHeight) * 0.7);
    if (side > 0) {
        target.style.width = `${side}px`;
        target.style.height = `${side}px`;
    }
}


function observeScannerTarget(video) {
    if (scannerTargetResizeObserver) scannerTargetResizeObserver.disconnect();
    updateScannerTarget(video);
    if (typeof ResizeObserver === "undefined") return;
    scannerTargetResizeObserver = new ResizeObserver(() => updateScannerTarget(video));
    scannerTargetResizeObserver.observe(video);
}


function openScanner() {
    document.getElementById("homePage").style.display = "none";
    document.getElementById("toolListPage").style.display = "none";
    const scanner = document.getElementById("scannerModal");
    scanner.style.display = "flex";
    scanner.setAttribute("aria-hidden", "false");
    observeScannerTarget(document.getElementById("cameraVideo"));
    updateScanResult("Ready to scan", "Start the camera and point it at a tool or piece of equipment. Matching uses the catalog images on this device.");
}


function closeScanner() {
    stopScanner();
    const scanner = document.getElementById("scannerModal");
    scanner.style.display = "none";
    scanner.setAttribute("aria-hidden", "true");
    document.getElementById("homePage").style.display = "flex";
}


async function prepareScannerBackend() {
    if (/Android/i.test(navigator.userAgent)) {
        if (tf.getBackend() !== "cpu" && !await tf.setBackend("cpu")) {
            throw new Error("This phone could not initialize the scanner's CPU compatibility mode.");
        }
        await tf.ready();
        return true;
    }

    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare the scanner image processor.");
    context.fillRect(0, 0, canvas.width, canvas.height);

    let pixels = null;
    let uploadError = null;
    try {
        pixels = tf.browser.fromPixels(canvas);
        pixels.dataSync();
    } catch (error) {
        uploadError = error;
    } finally {
        if (pixels) pixels.dispose();
    }

    if (!uploadError) return false;
    if (tf.getBackend() === "cpu") {
        throw new Error(`The scanner could not process images: ${uploadError.message || uploadError}`);
    }

    updateScanResult("Using compatibility mode", "This phone's GPU could not process scanner images. Switching to slower CPU processing.");
    if (!await tf.setBackend("cpu")) {
        throw new Error("This phone could not initialize the scanner's CPU compatibility mode.");
    }
    await tf.ready();

    let cpuPixels = null;
    try {
        cpuPixels = tf.browser.fromPixels(canvas);
        cpuPixels.dataSync();
    } catch (error) {
        throw new Error(`The scanner could not process images with either the phone's GPU or CPU: ${error.message || error}`);
    } finally {
        if (cpuPixels) cpuPixels.dispose();
    }
    return true;
}


async function requestScannerCamera() {
    const videoConstraints = {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
    };
    if (navigator.mediaDevices.getSupportedConstraints &&
        navigator.mediaDevices.getSupportedConstraints().focusMode) {
        videoConstraints.focusMode = { ideal: "continuous" };
    }

    try {
        return await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });
    } catch (error) {
        if (error.name === "NotAllowedError" || error.name === "SecurityError") throw error;
        return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
}


async function startScanner() {
    if (scannerStream) {
        stopScanner();
        updateScanResult("Scanner stopped", "Start the camera when you are ready to scan again.");
        return;
    }

    const button = document.getElementById("scannerToggle");
    const video = document.getElementById("cameraVideo");
    const generation = ++scannerGeneration;
    button.disabled = true;
    updateScanResult("Starting scanner", "Allow camera access. The recognition model is loaded from the internet; camera images stay on this device.");

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera access requires a supported browser and a secure connection (HTTPS).");
        }
        if (typeof tf === "undefined" || typeof mobilenet === "undefined") {
            throw new Error("The scanner recognition libraries did not load. Check your internet connection, close and reopen the app, then try again.");
        }

        const stream = await requestScannerCamera();
        if (generation !== scannerGeneration) {
            stream.getTracks().forEach((track) => track.stop());
            return;
        }
        scannerStream = stream;
        video.srcObject = stream;
        await video.play();
        observeScannerTarget(video);

        const usingCpuFallback = await prepareScannerBackend();
        try {
            scannerModelPromise = scannerModelPromise || mobilenet.load({ version: 2, alpha: 1.0 });
            scannerModel = scannerModel || await scannerModelPromise;
        } catch (error) {
            scannerModelPromise = null;
            throw new Error(`Could not load the recognition model. Check your internet connection and try again.${error.message ? ` (${error.message})` : ""}`);
        }
        if (generation !== scannerGeneration || !scannerStream) return;
        await prepareScannerIndex();
        if (generation !== scannerGeneration || !scannerStream) return;
        button.textContent = "Stop Scanner";
        document.getElementById("scannerTeachButton").disabled = false;
        const trainingNotice = scannerTrainingStorageError
            ? ` Saved examples could not be loaded: ${scannerTrainingStorageError} Built-in catalog matching is still available.`
            : "";
        const backendNotice = usingCpuFallback
            ? " This phone is using slower CPU compatibility mode because its GPU could not process scanner images."
            : "";
        updateScanResult("Scanning", `Center one item in the frame, hold steady, and give it a few seconds. Multiple zoom levels help with smaller or farther-away items.${backendNotice}${trainingNotice}`);
        scannerInterval = window.setInterval(scanCameraFrame, 1200);
    } catch (error) {
        if (generation !== scannerGeneration) return;
        if (!scannerModel) scannerModelPromise = null;
        stopScanner();
        updateScanResult("Scanner unavailable", error.message || "Could not start the camera or load the recognition model. Check camera permission and internet access, then try again.");
    } finally {
        button.disabled = false;
    }
}


function stopScanner() {
    scannerGeneration += 1;
    if (scannerTargetResizeObserver) {
        scannerTargetResizeObserver.disconnect();
        scannerTargetResizeObserver = null;
    }
    if (scannerInterval) {
        window.clearInterval(scannerInterval);
        scannerInterval = null;
    }

    if (scannerStream) {
        scannerStream.getTracks().forEach((track) => track.stop());
        scannerStream = null;
    }

    const video = document.getElementById("cameraVideo");
    if (video) video.srcObject = null;

    const button = document.getElementById("scannerToggle");
    if (button) button.textContent = "Start Scanner";
    const teachButton = document.getElementById("scannerTeachButton");
    if (teachButton) teachButton.disabled = true;
    const trainingSaveButton = document.getElementById("scannerTrainingSave");
    if (trainingSaveButton) trainingSaveButton.disabled = false;
    const trainingPanel = document.getElementById("scannerTraining");
    if (trainingPanel) trainingPanel.hidden = true;
    scannerBusy = false;
    lastRecognizedItemName = null;
    scannerEmbeddingHistory = [];
    scannerMatchHistory = [];

    const popup = document.getElementById("recognitionPopup");
    if (popup) {
        popup.style.display = "none";
        popup.setAttribute("aria-hidden", "true");
    }
    scannerPendingMatch = null;
}


async function scanCameraFrame() {
    const video = document.getElementById("cameraVideo");
    if (scannerBusy || !scannerModel || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    const generation = scannerGeneration;
    scannerBusy = true;
    try {
        const frameEmbeddings = [];
        let predictions = [];
        for (let index = 0; index < scannerFrameCropScales.length; index += 1) {
            const result = await analyzeScannerRegion(
                video,
                "camera frame",
                index === 0,
                scannerFrameCropScales[index]
            );
            frameEmbeddings.push(result.embedding);
            if (index === 0) predictions = result.predictions;
        }
        if (generation !== scannerGeneration || !scannerStream) return;
        const smoothedEmbeddings = smoothScannerEmbeddings(frameEmbeddings);
        const rankedMatches = scannerIndex
            .map((item) => {
                const similarity = getScannerItemSimilarity(item, smoothedEmbeddings);
                const labelEvidence = getScannerLabelEvidence(item, predictions, scannerIndex);
                return {
                    item,
                    similarity,
                    labelEvidence,
                    score: similarity + labelEvidence * 0.15
                };
            })
            .sort((a, b) => b.score - a.score);
        const match = rankedMatches[0];

        scannerMatchHistory.push(isScannerCatalogMatch(match, rankedMatches[1]) ? match : null);
        if (scannerMatchHistory.length > 4) scannerMatchHistory.shift();
        const confirmedMatch = getConfirmedScannerMatch(scannerMatchHistory);
        if (confirmedMatch && confirmedMatch.item.name !== lastRecognizedItemName) {
            lastRecognizedItemName = confirmedMatch.item.name;
            showRecognizedItem(confirmedMatch.item, confirmedMatch.similarity, smoothedEmbeddings);
        }
    } catch (error) {
        if (generation !== scannerGeneration) return;
        stopScanner();
        updateScanResult("Recognition stopped", error.message || "The camera frame could not be analyzed.");
    } finally {
        if (generation === scannerGeneration) scannerBusy = false;
    }
}


function showRecognizedItem(item, similarity, embeddings) {
    if (scannerInterval) {
        window.clearInterval(scannerInterval);
        scannerInterval = null;
    }

    scannerPendingMatch = { item, embeddings };
    document.getElementById("recognitionFeedback").textContent = "Is this the correct item? Your feedback trains the scanner on this device only.";
    document.getElementById("recognitionTrainingStatus").textContent = "";
    document.getElementById("recognitionActions").hidden = false;
    document.getElementById("recognitionCorrection").hidden = true;

    renderRecognizedItem(item, similarity);

    const popup = document.getElementById("recognitionPopup");
    popup.style.display = "flex";
    popup.setAttribute("aria-hidden", "false");
}


function renderRecognizedItem(item, similarity) {
    const similarityText = similarity === null
        ? "Corrected by your selection"
        : `Visual similarity score: ${similarity.toFixed(3)}`;
    const result = document.getElementById("scanResult");
    result.classList.add("scan-success");
    result.innerHTML = `
        <div class="scan-preview">
            <img src="${item.image}" alt="${item.name}">
        </div>
        <h4>${item.name}</h4>
        <p><strong>${similarityText}</strong></p>
        <p><strong>Function:</strong> ${item.function}</p>
        <p><strong>Uses:</strong> ${item.explanation}</p>
    `;

    document.getElementById("recognitionImage").src = item.image;
    document.getElementById("recognitionImage").alt = item.name;
    document.getElementById("recognitionName").textContent = item.name;
    document.getElementById("recognitionSimilarity").textContent = similarityText;
    document.getElementById("recognitionFunction").textContent = item.function;
    document.getElementById("recognitionExplanation").textContent = item.explanation;
}


function confirmScannerMatch() {
    if (!scannerPendingMatch) return;
    try {
        rememberScannerExample(scannerPendingMatch.item, scannerPendingMatch.embeddings);
        document.getElementById("recognitionTrainingStatus").textContent =
            "Saved. This confirmed camera view will improve future scans on this device.";
        document.getElementById("recognitionActions").hidden = true;
    } catch (error) {
        document.getElementById("recognitionTrainingStatus").textContent =
            error.message || "Could not save this example.";
    }
}


function showScannerCorrectionOptions() {
    if (!scannerPendingMatch) return;
    const select = document.getElementById("recognitionCorrectionSelect");
    select.textContent = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose the correct catalog item";
    select.appendChild(placeholder);

    scannerIndex.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        select.appendChild(option);
    });

    document.getElementById("recognitionFeedback").textContent =
        "Choose the right item; the current camera view will be saved as its example.";
    document.getElementById("recognitionActions").hidden = true;
    document.getElementById("recognitionCorrection").hidden = false;
    document.getElementById("recognitionTrainingStatus").textContent = "";
}


function confirmScannerCorrection() {
    if (!scannerPendingMatch) return;
    const selectedName = document.getElementById("recognitionCorrectionSelect").value;
    const correctedItem = scannerIndex.find((item) => item.name === selectedName);
    if (!correctedItem) {
        document.getElementById("recognitionTrainingStatus").textContent =
            "Select the correct catalog item first.";
        return;
    }

    try {
        rememberScannerExample(correctedItem, scannerPendingMatch.embeddings);
        scannerPendingMatch.item = correctedItem;
        renderRecognizedItem(correctedItem, null);
        document.getElementById("recognitionFeedback").textContent = "Thanks for correcting the scanner.";
        document.getElementById("recognitionCorrection").hidden = true;
        document.getElementById("recognitionTrainingStatus").textContent =
            "Saved. This camera view will help identify this item in future scans on this device.";
    } catch (error) {
        document.getElementById("recognitionTrainingStatus").textContent =
            error.message || "Could not save this example.";
    }
}


function closeRecognitionPopup() {
    const popup = document.getElementById("recognitionPopup");
    popup.style.display = "none";
    popup.setAttribute("aria-hidden", "true");
    scannerPendingMatch = null;

    if (scannerStream && !scannerInterval) {
        scannerEmbeddingHistory = [];
        scannerMatchHistory = [];
        scannerInterval = window.setInterval(scanCameraFrame, 1200);
    }
}


function updateScanResult(title, message) {
    const result = document.getElementById("scanResult");
    result.classList.remove("scan-success");
    result.innerHTML = `<h4>${title}</h4><p>${message}</p>`;
}


function filterCatalogCards(query) {
    const terms = normalizeScannerText(query).split(" ").filter(Boolean);
    const sections = [
        { id: "toolsSection", emptyId: "toolsSearchEmpty" },
        { id: "equipmentSection", emptyId: "equipmentSearchEmpty" }
    ];

    sections.forEach((sectionInfo) => {
        const section = document.getElementById(sectionInfo.id);
        const cards = section.querySelectorAll(".tool-card");
        let visibleCount = 0;
        cards.forEach((card) => {
            const image = card.querySelector("img");
            const searchableText = normalizeScannerText([
                card.textContent,
                image ? image.alt : "",
                card.getAttribute("data-scan-labels") || ""
            ].join(" "));
            const matches = terms.every((term) => searchableText.includes(term));
            card.hidden = !matches;
            if (matches) visibleCount += 1;
        });

        document.getElementById(sectionInfo.emptyId).hidden = visibleCount > 0;
    });
}


function openToolList(category = "all") {
    stopScanner();
    document.getElementById("scannerModal").style.display = "none";
    document.getElementById("homePage").style.display = "none";
    document.getElementById("toolListPage").style.display = "block";
    const searchInput = document.getElementById("catalogSearch");
    searchInput.value = "";
    filterCatalogCards("");

    const toolsSection = document.getElementById("toolsSection");
    const equipmentSection = document.getElementById("equipmentSection");
    const headerTitle = document.getElementById("listHeaderTitle");
    const headerSubtitle = document.getElementById("listHeaderSubtitle");

    if (category === "equipment") {
        toolsSection.style.display = "none";
        equipmentSection.style.display = "block";
        headerTitle.textContent = "EQUIPMENT";
        headerSubtitle.textContent = "Networking Equipment";
    } else {
        toolsSection.style.display = "block";
        equipmentSection.style.display = "none";
        headerTitle.textContent = "TOOL";
        headerSubtitle.textContent = "Networking Tools";
    }

    const activeSection = category === "equipment" ? equipmentSection : toolsSection;
    setTimeout(() => {
        if (activeSection) activeSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
}


function goHome() {
    stopScanner();
    document.getElementById("scannerModal").style.display = "none";
    document.getElementById("toolListPage").style.display = "none";
    document.getElementById("homePage").style.display = "flex";

    const toolsSection = document.getElementById("toolsSection");
    const equipmentSection = document.getElementById("equipmentSection");
    if (toolsSection) toolsSection.style.display = "block";
    if (equipmentSection) equipmentSection.style.display = "block";

    const headerTitle = document.getElementById("listHeaderTitle");
    const headerSubtitle = document.getElementById("listHeaderSubtitle");
    if (headerTitle) headerTitle.textContent = "TOOL & EQUIPMENT";
    if (headerSubtitle) headerSubtitle.textContent = "Networking Tools & Equipment";
    const searchInput = document.getElementById("catalogSearch");
    if (searchInput) {
        searchInput.value = "";
        filterCatalogCards("");
    }
}


function openTutorial() {
    const tutorial = document.getElementById("tutorialModal");
    tutorial.style.display = "flex";
    tutorial.setAttribute("aria-hidden", "false");
    document.getElementById("tutorialClose").focus();
}


function closeTutorial() {
    const tutorial = document.getElementById("tutorialModal");
    tutorial.style.display = "none";
    tutorial.setAttribute("aria-hidden", "true");
    const openButton = document.querySelector(".tutorial-open");
    if (openButton) openButton.focus();
}