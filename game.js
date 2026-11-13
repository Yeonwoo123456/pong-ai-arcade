
"use strict";

/* =========================================================
   PONG AI - FINAL STABLE VERSION
   Initial prediction error: 100% (large offset)
   AI tracking: heavily reduced
   Growth: Every completed point
   TensorFlow.js: Optional
   ========================================================= */

const CONFIG = {
    width: 900,
    height: 620,

    ballRadius: 9,
    ballSpeed: 5,

    playerSpeed: 8,
    playerWidth: 125,
    paddleHeight: 13,

    aiBaseSpeed: 1.3,
    aiMaxSpeed: 4.2,

    aiBaseWidth: 42,
    aiMaxWidth: 75,

    initialErrorRate: 1.00,
    finalErrorRate: 0.15,

    initialErrorRange: 480,
    finalErrorRange: 70,

    maxLevel: 30,

    replayCapacity: 5000,
    batchSize: 32,
    gamma: 0.95,
    learningRate: 0.0005,

    epsilonStart: 0.35,
    epsilonMin: 0.03,
    epsilonDecay: 0.995,

    trainingBatchesPerPoint: 2,
    targetUpdateFrequency: 10,
    saveFrequency: 10
};

const STORAGE = {
    model: "pong-dqn-model-v9",
    stats: "pong-statistics",
    attempts: "pong-ai-attempts-v9"
};

const $ = id => document.getElementById(id);

const ui = {
    menu: $("menuScreen"),
    game: $("gameScreen"),
    stats: $("statsScreen"),
    arena: $("arena"),

    play: $("playBtn"),
    statsButton: $("statsBtn"),
    statsBack: $("statsBackBtn"),
    gameMenu: $("gameMenuBtn"),
    pauseMenu: $("pauseMenuBtn"),
    endMatch: $("endMatchBtn"),
    newMatch: $("newMatchBtn"),

    pause: $("pauseBtn"),
    resume: $("resumeBtn"),
    pauseOverlay: $("pauseOverlay"),

    resetStats: $("resetStatsBtn"),

    modelStatus: $("modelStatus"),

    liveRate: $("liveRate"),
    liveUpdates: $("liveUpdates"),
    liveSpeed: $("liveSpeed"),

    aiScoreDisplay: $("aiScoreDisplay"),
    playerScoreDisplay: $("playerScoreDisplay"),

    statMatches: $("statMatches"),
    statRate: $("statRate"),
    statTime: $("statTime"),
    statUpdates: $("statUpdates"),
    statExperiences: $("statExperiences"),
    statEpsilon: $("statEpsilon"),

    chart: $("statsChart"),
    tooltip: $("chartTooltip")
};

const canvas = ui.arena;
const ctx = canvas ? canvas.getContext("2d") : null;

function storageGet(key, fallback = null) {
    try {
        const value = localStorage.getItem(key);
        return value === null ? fallback : value;
    } catch (error) {
        console.warn("Storage read failed:", error);
        return fallback;
    }
}

function storageSet(key, value) {
    try {
        localStorage.setItem(key, value);
        return true;
    } catch (error) {
        console.warn("Storage write failed:", error);
        return false;
    }
}

function loadNumber(key, fallback) {
    const value = Number(storageGet(key, fallback));
    return Number.isFinite(value) ? value : fallback;
}

let running = false;
let paused = false;
let animationFrame = null;

let playerScore = 0;
let aiScore = 0;

let matchStartTime = 0;

let aiAttempts = Math.max(
    0,
    loadNumber(STORAGE.attempts, 0)
);

let aiLevel = 1;
let aiSkill = 0;

let agent = null;
let trainingInProgress = false;
let trainingQueued = false;

let totalUpdates = 0;
let totalExperiences = 0;

let keys = {
    left: false,
    right: false
};

let statistics = {
    matches: 0,
    playerPoints: 0,
    aiPoints: 0,
    totalDuration: 0,
    history: []
};

let player = {
    x: CONFIG.width / 2,
    y: CONFIG.height - 45,
    width: CONFIG.playerWidth,
    height: CONFIG.paddleHeight
};

let ai = {
    x: CONFIG.width / 2,
    y: 35,
    width: CONFIG.aiBaseWidth,
    height: CONFIG.paddleHeight
};

let ball = {
    x: CONFIG.width / 2,
    y: CONFIG.height / 2,
    vx: 3,
    vy: -CONFIG.ballSpeed,
    radius: CONFIG.ballRadius
};

let servePredictionFails = true;
let servePredictionOffset = 0;

let lastAIState = null;
let lastAIAction = 1;
let pointFinished = false;

let audioContext = null;

function playSound(frequency = 440, duration = 0.06) {
    try {
        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;

        if (!AudioContextClass) return;

        if (!audioContext) {
            audioContext = new AudioContextClass();
        }

        if (audioContext.state === "suspended") {
            audioContext.resume().catch(() => {});
        }

        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();

        oscillator.type = "square";
        oscillator.frequency.value = frequency;

        gain.gain.setValueAtTime(
            0.04,
            audioContext.currentTime
        );

        gain.gain.exponentialRampToValueAtTime(
            0.001,
            audioContext.currentTime + duration
        );

        oscillator.connect(gain);
        gain.connect(audioContext.destination);

        oscillator.start();
        oscillator.stop(
            audioContext.currentTime + duration
        );
    } catch (error) {
        console.warn("Audio unavailable:", error);
    }
}

function loadStatistics() {
    try {
        const saved = storageGet(STORAGE.stats);
        if (!saved) return;

        const parsed = JSON.parse(saved);

        statistics = {
            ...statistics,
            ...parsed,
            history: Array.isArray(parsed.history)
                ? parsed.history
                : []
        };
    } catch (error) {
        console.warn("Statistics load failed:", error);
    }
}

function saveStatistics() {
    storageSet(
        STORAGE.stats,
        JSON.stringify(statistics)
    );
}

function updateAISkill() {
    aiLevel = Math.min(
        CONFIG.maxLevel,
        aiAttempts + 1
    );

    aiSkill = Math.min(
        1,
        aiAttempts / (CONFIG.maxLevel - 1)
    );
}

function getAISpeed() {
    return CONFIG.aiBaseSpeed +
        (CONFIG.aiMaxSpeed - CONFIG.aiBaseSpeed) *
        aiSkill;
}

function getAIWidth() {
    return CONFIG.aiBaseWidth +
        (CONFIG.aiMaxWidth - CONFIG.aiBaseWidth) *
        aiSkill;
}

function getAIErrorRate() {
    return CONFIG.initialErrorRate +
        (CONFIG.finalErrorRate - CONFIG.initialErrorRate) *
        aiSkill;
}

function getAIErrorRange() {
    return CONFIG.initialErrorRange +
        (CONFIG.finalErrorRange - CONFIG.initialErrorRange) *
        aiSkill;
}

function getAIPredictionAccuracy() {
    return 1 - getAIErrorRate();
}

function advanceAISkill() {
    aiAttempts++;
    updateAISkill();

    storageSet(
        STORAGE.attempts,
        String(aiAttempts)
    );

    updateUI();
}

class ReplayMemory {
    constructor(capacity) {
        this.capacity = capacity;
        this.data = [];
        this.position = 0;
    }

    add(item) {
        if (this.data.length < this.capacity) {
            this.data.push(item);
        } else {
            this.data[this.position] = item;
        }

        this.position =
            (this.position + 1) % this.capacity;
    }

    sample(count) {
        const result = [];
        const size = Math.min(count, this.data.length);

        for (let i = 0; i < size; i++) {
            result.push(
                this.data[
                    Math.floor(Math.random() * this.data.length)
                ]
            );
        }

        return result;
    }

    get length() {
        return this.data.length;
    }
}

class DQNAgent {
    constructor(tf) {
        this.tf = tf;
        this.memory = new ReplayMemory(
            CONFIG.replayCapacity
        );

        this.model = null;
        this.targetModel = null;

        this.epsilon = CONFIG.epsilonStart;
        this.updates = 0;
        this.experiences = 0;
        this.ready = false;
    }

    createModel() {
        const model = this.tf.sequential();

        model.add(this.tf.layers.dense({
            inputShape: [5],
            units: 64,
            activation: "relu"
        }));

        model.add(this.tf.layers.dense({
            units: 64,
            activation: "relu"
        }));

        model.add(this.tf.layers.dense({
            units: 3,
            activation: "linear"
        }));

        model.compile({
            optimizer: this.tf.train.adam(
                CONFIG.learningRate
            ),
            loss: "meanSquaredError"
        });

        return model;
    }

    async initialize() {
        this.model = this.createModel();
        this.targetModel = this.createModel();

        try {
            const loaded = await this.tf.loadLayersModel(
                `localstorage://${STORAGE.model}`
            );

            this.model.dispose();
            this.model = loaded;

            this.copyWeights();

            console.log("Saved AI model loaded.");
        } catch (error) {
            console.log("Starting a new AI model.");
        }

        this.ready = true;
    }

    copyWeights() {
        const weights = this.model.getWeights();
        const copies = weights.map(w => w.clone());

        this.targetModel.setWeights(copies);

        copies.forEach(w => w.dispose());
    }

    chooseAction(state) {
        if (!this.ready) return 1;

        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 3);
        }

        return this.tf.tidy(() => {
            const input = this.tf.tensor2d([state]);
            const output = this.model.predict(input);

            return output.argMax(1).dataSync()[0];
        });
    }

    remember(state, action, reward, nextState, done) {
        if (!state || !nextState) return;

        this.memory.add({
            state: [...state],
            action,
            reward,
            nextState: [...nextState],
            done
        });

        this.experiences++;
        totalExperiences++;
    }

    async learnOnce() {
        if (
            !this.ready ||
            this.memory.length < CONFIG.batchSize
        ) {
            return;
        }

        const tf = this.tf;
        const batch = this.memory.sample(CONFIG.batchSize);

        const states = batch.map(item => item.state);
        const nextStates = batch.map(item => item.nextState);

        const stateTensor = tf.tensor2d(states);
        const nextTensor = tf.tensor2d(nextStates);

        let currentQ = null;
        let nextQ = null;
        let targetTensor = null;

        try {
            currentQ = this.model.predict(stateTensor);
            nextQ = this.targetModel.predict(nextTensor);

            const currentValues = await currentQ.array();
            const nextValues = await nextQ.array();

            const targets = currentValues.map((values, i) => {
                const updated = [...values];
                const future = Math.max(...nextValues[i]);

                updated[batch[i].action] = batch[i].done
                    ? batch[i].reward
                    : batch[i].reward +
                        CONFIG.gamma * future;

                return updated;
            });

            targetTensor = tf.tensor2d(targets);

            await this.model.fit(
                stateTensor,
                targetTensor,
                {
                    epochs: 1,
                    batchSize: CONFIG.batchSize,
                    shuffle: true,
                    verbose: 0
                }
            );

            this.updates++;
            totalUpdates++;

            this.epsilon = Math.max(
                CONFIG.epsilonMin,
                this.epsilon * CONFIG.epsilonDecay
            );

            if (
                this.updates %
                CONFIG.targetUpdateFrequency === 0
            ) {
                this.copyWeights();
            }

            updateUI();

        } finally {
            if (currentQ) currentQ.dispose();
            if (nextQ) nextQ.dispose();
            if (targetTensor) targetTensor.dispose();

            stateTensor.dispose();
            nextTensor.dispose();
        }
    }

    async trainAfterPoint() {
        if (!this.ready) return;

        if (trainingInProgress) {
            trainingQueued = true;
            return;
        }

        trainingInProgress = true;

        try {
            do {
                trainingQueued = false;

                for (
                    let i = 0;
                    i < CONFIG.trainingBatchesPerPoint;
                    i++
                ) {
                    if (
                        this.memory.length <
                        CONFIG.batchSize
                    ) {
                        break;
                    }

                    await this.learnOnce();
                }

                if (
                    this.updates > 0 &&
                    this.updates % CONFIG.saveFrequency === 0
                ) {
                    await this.save();
                }

            } while (trainingQueued);

        } catch (error) {
            console.error("DQN training failed:", error);

        } finally {
            trainingInProgress = false;
        }
    }

    async save() {
        if (!this.ready || !this.model) return;

        try {
            await this.model.save(
                `localstorage://${STORAGE.model}`
            );
        } catch (error) {
            console.warn("Model save failed:", error);
        }
    }
}

function getState() {
    return [
        ball.x / CONFIG.width,
        ball.y / CONFIG.height,
        ball.vx / 10,
        ball.vy / 10,
        (ai.x - ball.x) / CONFIG.width
    ];
}

function predictBallXAtY(targetY) {
    if (ball.vy >= 0) return ball.x;

    const time = (targetY - ball.y) / ball.vy;

    if (time <= 0) return ball.x;

    const projected = ball.x + ball.vx * time;

    const left = CONFIG.ballRadius;
    const right = CONFIG.width - CONFIG.ballRadius;
    const range = right - left;

    let relative = (projected - left) % (range * 2);

    if (relative < 0) {
        relative += range * 2;
    }

    if (relative > range) {
        relative = range * 2 - relative;
    }

    return left + relative;
}

function moveAI() {
    ai.width = getAIWidth();

    moveAI.frame = (moveAI.frame || 0) + 1;

    const reactionFrames = Math.max(
        8,
        Math.round(18 - 10 * aiSkill)
    );

    if (
        moveAI.frame % reactionFrames === 0 ||
        moveAI.cachedTargetX === undefined
    ) {
        if (ball.vy < 0) {
            const prediction = predictBallXAtY(ai.y);

            const errorOffset = servePredictionFails
                ? servePredictionOffset
                : 0;

            moveAI.cachedTargetX = prediction + errorOffset;

            if (agent && agent.ready) {
                const state = getState();
                const action = agent.chooseAction(state);

                lastAIState = [...state];
                lastAIAction = action;

                const influence = Math.min(
                    0.03,
                    agent.updates / 5000
                );

                moveAI.cachedTargetX +=
                    (action - 1) * getAISpeed() * influence;
            }
        } else {
            moveAI.cachedTargetX = CONFIG.width / 2;
        }
    }

    const minX = ai.width / 2;
    const maxX = CONFIG.width - ai.width / 2;

    const targetX = Math.max(
        minX,
        Math.min(maxX, moveAI.cachedTargetX)
    );

    const difference = targetX - ai.x;

    const trackingStrength = 0.08 + 0.08 * aiSkill;
    const maxMovement = getAISpeed() * 0.35;

    const movement = Math.max(
        -maxMovement,
        Math.min(
            maxMovement,
            difference * trackingStrength
        )
    );

    if (Math.abs(difference) > 8) {
        ai.x += movement;
    }
}

function recordExperience(reward, done) {
    if (!agent || !agent.ready || !lastAIState) {
        return;
    }

    agent.remember(
        lastAIState,
        lastAIAction,
        reward,
        getState(),
        done
    );

    lastAIState = null;
}

function resetBall() {
    ball.x = CONFIG.width / 2;
    ball.y = CONFIG.height / 2;

    ball.vx =
        (Math.random() < 0.5 ? -1 : 1) *
        (2 + Math.random() * 2);

    ball.vy =
        (Math.random() < 0.5 ? -1 : 1) *
        CONFIG.ballSpeed;

    servePredictionFails =
        Math.random() < getAIErrorRate();

    const errorDirection = Math.random() < 0.5 ? -1 : 1;

    const errorMagnitude =
        getAIErrorRange() * (0.65 + Math.random() * 0.35);

    servePredictionOffset = errorDirection * errorMagnitude;

    lastAIState = null;
    lastAIAction = 1;

    moveAI.cachedTargetX = CONFIG.width / 2;
}

function ballHitsPaddle(paddle) {
    return (
        ball.x + ball.radius >
            paddle.x - paddle.width / 2 &&
        ball.x - ball.radius <
            paddle.x + paddle.width / 2 &&
        ball.y + ball.radius >
            paddle.y - paddle.height / 2 &&
        ball.y - ball.radius <
            paddle.y + paddle.height / 2
    );
}

function bounceFromPaddle(paddle, isAI) {
    const relative = Math.max(
        -1,
        Math.min(
            1,
            (ball.x - paddle.x) / (paddle.width / 2)
        )
    );

    ball.vx = relative * 5;

    ball.vy = isAI
        ? Math.abs(ball.vy)
        : -Math.abs(ball.vy);

    ball.vy *= 1.02;

    if (isAI) {
        recordExperience(1, false);
    }

    playSound(isAI ? 520 : 650);
}

function finishPoint(playerWon) {
    if (pointFinished) return;

    pointFinished = true;

    if (playerWon) {
        playerScore++;
        statistics.playerPoints++;

        recordExperience(-1, true);
        playSound(260, 0.15);

    } else {
        aiScore++;
        statistics.aiPoints++;

        recordExperience(1.5, true);
        playSound(800, 0.15);
    }

    advanceAISkill();

    if (agent && agent.ready) {
        agent.trainAfterPoint();
    }

    resetBall();

    pointFinished = false;

    updateUI();
}

function moveGame() {
    if (!running || paused) return;

    if (keys.left) {
        player.x -= CONFIG.playerSpeed;
    }

    if (keys.right) {
        player.x += CONFIG.playerSpeed;
    }

    player.x = Math.max(
        player.width / 2,
        Math.min(
            CONFIG.width - player.width / 2,
            player.x
        )
    );

    moveAI();

    ball.x += ball.vx;
    ball.y += ball.vy;

    if (
        ball.x - ball.radius <= 0 ||
        ball.x + ball.radius >= CONFIG.width
    ) {
        ball.vx *= -1;
        playSound(350);
    }

    if (ball.vy < 0 && ballHitsPaddle(ai)) {
        ball.y =
            ai.y + ai.height / 2 + ball.radius;

        bounceFromPaddle(ai, true);
    }

    if (ball.vy > 0 && ballHitsPaddle(player)) {
        ball.y =
            player.y - player.height / 2 -
            ball.radius;

        bounceFromPaddle(player, false);
    }

    if (ball.y - ball.radius > CONFIG.height) {
        finishPoint(true);
    }

    if (ball.y + ball.radius < 0) {
        finishPoint(false);
    }

    if (ui.aiScoreDisplay) {
        ui.aiScoreDisplay.textContent = String(aiScore);
    }

    if (ui.playerScoreDisplay) {
        ui.playerScoreDisplay.textContent = String(playerScore);
    }
}

function drawGame() {
    if (!ctx) return;

    ctx.clearRect(0, 0, CONFIG.width, CONFIG.height);

    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, CONFIG.width, CONFIG.height);

    ctx.strokeStyle = "#444";
    ctx.setLineDash([10, 12]);

    ctx.beginPath();
    ctx.moveTo(0, CONFIG.height / 2);
    ctx.lineTo(CONFIG.width, CONFIG.height / 2);
    ctx.stroke();

    ctx.setLineDash([]);

    ctx.fillStyle = "#b6ff00";
    ctx.fillRect(
        ai.x - ai.width / 2,
        ai.y - ai.height / 2,
        ai.width,
        ai.height
    );

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(
        player.x - player.width / 2,
        player.y - player.height / 2,
        player.width,
        player.height
    );

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(
        ball.x,
        ball.y,
        ball.radius,
        0,
        Math.PI * 2
    );
    ctx.fill();
}

function gameLoop() {
    if (!running) return;

    try {
        if (!paused) {
            moveGame();
            drawGame();
        }
    } catch (error) {
        console.error("Game loop error:", error);

        running = false;
        showScreen("menu");

        if (ui.modelStatus) {
            ui.modelStatus.textContent =
                "GAME ERROR - CHECK CONSOLE";
        }

        return;
    }

    animationFrame = requestAnimationFrame(gameLoop);
}

function showScreen(name) {
    if (ui.menu) {
        ui.menu.style.display =
            name === "menu" ? "flex" : "none";
    }

    if (ui.game) {
        ui.game.style.display =
            name === "game" ? "flex" : "none";
    }

    if (ui.stats) {
        ui.stats.style.display =
            name === "stats" ? "flex" : "none";
    }
}

function startMatch() {
    playerScore = 0;
    aiScore = 0;

    if (ui.aiScoreDisplay) {
        ui.aiScoreDisplay.textContent = "0";
    }

    if (ui.playerScoreDisplay) {
        ui.playerScoreDisplay.textContent = "0";
    }

    player.x = CONFIG.width / 2;
    ai.x = CONFIG.width / 2;

    matchStartTime = Date.now();

    resetBall();

    running = true;
    paused = false;

    if (ui.pauseOverlay) {
        ui.pauseOverlay.style.display = "none";
    }

    showScreen("game");

    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }

    gameLoop();
}

function pauseGame() {
    if (!running) return;

    paused = true;

    if (ui.pauseOverlay) {
        ui.pauseOverlay.style.display = "flex";
    }
}

function resumeGame() {
    if (!running) return;

    paused = false;

    if (ui.pauseOverlay) {
        ui.pauseOverlay.style.display = "none";
    }
}

async function endMatch() {
    if (!running) return;

    running = false;
    paused = false;

    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }

    const duration = Math.floor(
        (Date.now() - matchStartTime) / 1000
    );

    statistics.matches++;
    statistics.totalDuration += duration;

    statistics.history.push({
        playerScore,
        aiScore,
        duration,
        aiLevel,
        date: new Date().toISOString()
    });

    if (statistics.history.length > 100) {
        statistics.history.shift();
    }

    saveStatistics();

    if (agent && agent.ready) {
        await agent.save();
    }

    showScreen("menu");
    updateUI();
}

function getWinRate() {
    const total =
        statistics.playerPoints +
        statistics.aiPoints;

    if (total === 0) return 0;

    return Math.round(
        statistics.playerPoints / total * 100
    );
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;

    return `${minutes}m ${remaining}s`;
}

function updateUI() {
    updateAISkill();

    const accuracy = Math.round(
        getAIPredictionAccuracy() * 100
    );

    if (ui.liveRate) {
        ui.liveRate.textContent = `${accuracy}%`;
    }

    if (ui.liveUpdates) {
        ui.liveUpdates.textContent = String(aiAttempts);
    }

    if (ui.liveSpeed) {
        ui.liveSpeed.textContent = `LV.${aiLevel}`;
    }

    if (ui.aiScoreDisplay) {
        ui.aiScoreDisplay.textContent = String(aiScore);
    }

    if (ui.playerScoreDisplay) {
        ui.playerScoreDisplay.textContent = String(playerScore);
    }

    if (ui.statMatches) {
        ui.statMatches.textContent = statistics.matches;
    }

    if (ui.statRate) {
        ui.statRate.textContent = `${getWinRate()}%`;
    }

    if (ui.statTime) {
        ui.statTime.textContent =
            formatTime(statistics.totalDuration);
    }

    if (ui.statUpdates) {
        ui.statUpdates.textContent =
            agent ? agent.updates : totalUpdates;
    }

    if (ui.statExperiences) {
        ui.statExperiences.textContent =
            agent ? agent.experiences : totalExperiences;
    }

    if (ui.statEpsilon) {
        ui.statEpsilon.textContent =
            agent ? agent.epsilon.toFixed(3) : "N/A";
    }
}

function drawChart() {
    const chart = ui.chart;
    if (!chart) return;

    const chartCtx = chart.getContext("2d");
    if (!chartCtx) return;

    const width = chart.width;
    const height = chart.height;

    chartCtx.clearRect(0, 0, width, height);

    chartCtx.fillStyle = "#080808";
    chartCtx.fillRect(0, 0, width, height);

    const history = statistics.history;

    if (!history.length) {
        chartCtx.fillStyle = "#888";
        chartCtx.font = "14px monospace";
        chartCtx.textAlign = "center";

        chartCtx.fillText(
            "NO MATCH DATA",
            width / 2,
            height / 2
        );

        return;
    }

    const padding = 35;
    const graphWidth = width - padding * 2;
    const graphHeight = height - padding * 2;

    const maxScore = Math.max(
        1,
        ...history.map(item =>
            Math.max(item.playerScore, item.aiScore)
        )
    );

    chartCtx.strokeStyle = "#444";

    for (let i = 0; i <= 4; i++) {
        const y = padding + graphHeight * i / 4;

        chartCtx.beginPath();
        chartCtx.moveTo(padding, y);
        chartCtx.lineTo(width - padding, y);
        chartCtx.stroke();
    }

    function drawLine(key, color) {
        chartCtx.beginPath();
        chartCtx.strokeStyle = color;
        chartCtx.lineWidth = 2;

        history.forEach((item, index) => {
            const x = padding +
                (
                    history.length === 1
                        ? graphWidth / 2
                        : graphWidth * index /
                            (history.length - 1)
                );

            const y = height - padding -
                item[key] / maxScore * graphHeight;

            if (index === 0) {
                chartCtx.moveTo(x, y);
            } else {
                chartCtx.lineTo(x, y);
            }
        });

        chartCtx.stroke();
    }

    drawLine("playerScore", "#ffffff");
    drawLine("aiScore", "#b6ff00");
}

function setupChartHover() {
    if (!ui.chart || !ui.tooltip) return;

    ui.chart.addEventListener("mousemove", event => {
        const history = statistics.history;

        if (!history.length) return;

        const rect = ui.chart.getBoundingClientRect();

        const ratio =
            (event.clientX - rect.left) / rect.width;

        const index = Math.max(
            0,
            Math.min(
                history.length - 1,
                Math.round(ratio * (history.length - 1))
            )
        );

        const item = history[index];

        ui.tooltip.textContent =
            `Player: ${item.playerScore} | ` +
            `AI: ${item.aiScore} | ` +
            `AI Level: ${item.aiLevel}`;

        ui.tooltip.style.display = "block";
        ui.tooltip.style.left =
            `${event.clientX - rect.left + 10}px`;

        ui.tooltip.style.top =
            `${event.clientY - rect.top + 10}px`;
    });

    ui.chart.addEventListener("mouseleave", () => {
        ui.tooltip.style.display = "none";
    });
}

document.addEventListener("keydown", event => {
    const key = event.key.toLowerCase();

    if (key === "arrowleft" || key === "a") {
        keys.left = true;
    }

    if (key === "arrowright" || key === "d") {
        keys.right = true;
    }

    if (event.key === "Escape" && running) {
        if (paused) {
            resumeGame();
        } else {
            pauseGame();
        }
    }
});

document.addEventListener("keyup", event => {
    const key = event.key.toLowerCase();

    if (key === "arrowleft" || key === "a") {
        keys.left = false;
    }

    if (key === "arrowright" || key === "d") {
        keys.right = false;
    }
});

function configureScoreboard() {
    const scoreboard = document.querySelector(".scoreboard");
    if (!scoreboard) return;

    Object.assign(scoreboard.style, {
        width: "100%",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "row",
        justifyContent: "center",
        alignItems: "center",
        flexWrap: "nowrap",
        gap: "clamp(24px, 8vw, 80px)",
        textAlign: "center",
        marginLeft: "auto",
        marginRight: "auto"
    });

    scoreboard.querySelectorAll(".score-item").forEach(item => {
        Object.assign(item.style, {
            flex: "0 1 120px",
            minWidth: "0",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            textAlign: "center"
        });
    });

    scoreboard.querySelectorAll(".score-number").forEach(number => {
        Object.assign(number.style, {
            display: "block",
            width: "100%",
            textAlign: "center",
            lineHeight: "1.15",
            fontVariantNumeric: "tabular-nums"
        });
    });
}

function setupButtons() {
    if (ui.play) {
        ui.play.addEventListener("click", startMatch);
    }

    if (ui.newMatch) {
        ui.newMatch.addEventListener("click", startMatch);
    }

    if (ui.statsButton) {
        ui.statsButton.addEventListener("click", () => {
            showScreen("stats");
            updateUI();
            drawChart();
        });
    }

    if (ui.statsBack) {
        ui.statsBack.addEventListener(
            "click",
            () => showScreen("menu")
        );
    }

    if (ui.gameMenu) {
        ui.gameMenu.addEventListener("click", endMatch);
    }

    if (ui.pauseMenu) {
        ui.pauseMenu.addEventListener("click", endMatch);
    }

    if (ui.endMatch) {
        ui.endMatch.addEventListener("click", endMatch);
    }

    if (ui.pause) {
        ui.pause.addEventListener("click", pauseGame);
    }

    if (ui.resume) {
        ui.resume.addEventListener("click", resumeGame);
    }

    if (ui.resetStats) {
        ui.resetStats.addEventListener("click", () => {
            if (!confirm("Reset all statistics?")) return;

            statistics = {
                matches: 0,
                playerPoints: 0,
                aiPoints: 0,
                totalDuration: 0,
                history: []
            };

            saveStatistics();
            updateUI();
            drawChart();
        });
    }
}

async function initializeAI() {
    if (!window.tf) {
        console.warn("TensorFlow.js unavailable. Basic AI active.");

        if (ui.modelStatus) {
            ui.modelStatus.textContent = "BASIC AI MODE";
        }

        return;
    }

    try {
        const tf = window.tf;

        try {
            await tf.setBackend("cpu");
            await tf.ready();
        } catch (backendError) {
            console.warn("Backend setup warning:", backendError);
        }

        const newAgent = new DQNAgent(tf);

        await newAgent.initialize();

        agent = newAgent;

        if (ui.modelStatus) {
            ui.modelStatus.textContent = "AI MODEL READY";
        }

        updateUI();

    } catch (error) {
        console.error(
            "AI initialization failed. Basic AI remains active:",
            error
        );

        agent = null;

        if (ui.modelStatus) {
            ui.modelStatus.textContent = "BASIC AI MODE";
        }
    }
}

async function init() {
    try {
        if (!canvas || !ctx) {
            console.error("Missing #arena canvas.");

            if (ui.modelStatus) {
                ui.modelStatus.textContent = "CANVAS NOT FOUND";
            }

            return;
        }

        canvas.width = CONFIG.width;
        canvas.height = CONFIG.height;

        loadStatistics();
        updateAISkill();

        configureScoreboard();
        setupButtons();
        setupChartHover();

        showScreen("menu");

        updateUI();
        drawChart();

        initializeAI().catch(error => {
            console.error("AI startup error:", error);
        });

        console.log("Pong AI initialized.");
        console.log("AI attempts:", aiAttempts);
        console.log("AI level:", aiLevel);
        console.log(
            "Prediction error rate:",
            getAIErrorRate()
        );

    } catch (error) {
        console.error("Initialization error:", error);
        showScreen("menu");

        if (ui.modelStatus) {
            ui.modelStatus.textContent = "BASIC AI MODE";
        }
    }
}

if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        init,
        { once: true }
    );
} else {
    init();
}
