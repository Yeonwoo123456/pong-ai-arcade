
"use strict";

/* =========================================================
   PONG AI - Progressive DQN Edition
   Initial AI strength: approximately 30%
   Learning: after every point
   ========================================================= */

const CONFIG = {
    width: 900,
    height: 620,

    paddleWidth: 125,
    paddleHeight: 13,
    ballRadius: 9,

    playerSpeed: 8,
    ballSpeed: 5,

    // AI starts at approximately 30% of its former initial speed.
    aiBaseSpeed: 1.35,
    aiMaxSpeed: 9.5,

    // AI paddle grows as it learns.
    aiBasePaddleWidth: 42,
    aiMaxPaddleWidth: 125,

    // Large initial prediction error; decreases with progress.
    aiInitialError: 190,
    aiMinimumError: 5,

    // One level per completed point, up to the maximum.
    aiMaxLevel: 30,

    // DQN
    replayCapacity: 12000,
    batchSize: 32,
    gamma: 0.95,
    learningRate: 0.0005,

    epsilonStart: 0.35,
    epsilonMin: 0.03,
    epsilonDecay: 0.995,

    targetUpdateFrequency: 10,
    saveFrequency: 5,

    // Model influence gradually increases with training.
    learnedAIStart: 10,

    // Training workload after every point.
    trainingBatchesPerPoint: 4
};

const MODEL_KEY = "pong-dqn-model-v6";
const STATS_KEY = "pong-statistics";
const ATTEMPTS_KEY = "pong-ai-attempts";
const PROGRESS_KEY = "pong-ai-progress";

/* =========================================================
   DOM
   ========================================================= */

const $ = id => document.getElementById(id);

const canvas = $("arena");
const ctx = canvas ? canvas.getContext("2d") : null;

const elements = {
    menuScreen: $("menuScreen"),
    gameScreen: $("gameScreen"),
    statsScreen: $("statsScreen"),

    playBtn: $("playBtn"),
    statsBtn: $("statsBtn"),
    statsBackBtn: $("statsBackBtn"),
    gameMenuBtn: $("gameMenuBtn"),
    endMatchBtn: $("endMatchBtn"),
    newMatchBtn: $("newMatchBtn"),

    pauseBtn: $("pauseBtn"),
    resumeBtn: $("resumeBtn"),
    pauseOverlay: $("pauseOverlay"),

    modelStatus: $("modelStatus"),

    liveRate: $("liveRate"),
    liveUpdates: $("liveUpdates"),
    liveSpeed: $("liveSpeed"),

    statMatches: $("statMatches"),
    statRate: $("statRate"),
    statTime: $("statTime"),
    statUpdates: $("statUpdates"),
    statExperiences: $("statExperiences"),
    statEpsilon: $("statEpsilon"),

    statsChart: $("statsChart"),
    chartTooltip: $("chartTooltip")
};

/* =========================================================
   GAME STATE
   ========================================================= */

let screen = "menu";
let running = false;
let paused = false;
let animationFrame = null;

let agent = null;
let trainingInProgress = false;
let pendingTraining = false;

let aiAttempts = Number(localStorage.getItem(ATTEMPTS_KEY) || 0);
let aiLevel = 1;
let aiSkill = 0;

let matchStartTime = 0;
let matchDuration = 0;

let playerScore = 0;
let aiScore = 0;

let totalUpdates = 0;
let totalExperiences = 0;

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
    width: CONFIG.paddleWidth,
    height: CONFIG.paddleHeight
};

let ai = {
    x: CONFIG.width / 2,
    y: 35,
    width: CONFIG.aiBasePaddleWidth,
    height: CONFIG.paddleHeight
};

let ball = {
    x: CONFIG.width / 2,
    y: CONFIG.height / 2,
    vx: CONFIG.ballSpeed,
    vy: CONFIG.ballSpeed,
    radius: CONFIG.ballRadius
};

let keys = {
    left: false,
    right: false
};

let lastAIState = null;
let lastAIAction = 1;
let aiAimOffset = 0;
let frameCounter = 0;

/* =========================================================
   AUDIO
   ========================================================= */

let audioContext = null;

function playSound(frequency = 440, duration = 0.06) {
    try {
        if (!audioContext) {
            audioContext = new (
                window.AudioContext ||
                window.webkitAudioContext
            )();
        }

        if (audioContext.state === "suspended") {
            audioContext.resume();
        }

        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();

        oscillator.type = "square";
        oscillator.frequency.value = frequency;

        gain.gain.setValueAtTime(0.045, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(
            0.001,
            audioContext.currentTime + duration
        );

        oscillator.connect(gain);
        gain.connect(audioContext.destination);

        oscillator.start();
        oscillator.stop(audioContext.currentTime + duration);
    } catch (error) {
        console.warn("Audio unavailable:", error);
    }
}

/* =========================================================
   AI PROGRESSION
   ========================================================= */

function updateAISkill() {
    aiLevel = Math.min(
        CONFIG.aiMaxLevel,
        Math.max(1, aiAttempts + 1)
    );

    aiSkill = Math.min(
        1,
        aiAttempts / (CONFIG.aiMaxLevel - 1)
    );
}

function getAISpeed() {
    return CONFIG.aiBaseSpeed +
        (CONFIG.aiMaxSpeed - CONFIG.aiBaseSpeed) * aiSkill;
}

function getAIPaddleWidth() {
    return CONFIG.aiBasePaddleWidth +
        (CONFIG.aiMaxPaddleWidth - CONFIG.aiBasePaddleWidth) * aiSkill;
}

function getAIAimError() {
    return CONFIG.aiInitialError +
        (CONFIG.aiMinimumError - CONFIG.aiInitialError) * aiSkill;
}

function advanceAISkill() {
    aiAttempts++;

    updateAISkill();

    localStorage.setItem(ATTEMPTS_KEY, String(aiAttempts));

    localStorage.setItem(PROGRESS_KEY, JSON.stringify({
        attempts: aiAttempts,
        level: aiLevel,
        skill: aiSkill
    }));

    updateGameUI();
}

/* =========================================================
   REPLAY MEMORY
   ========================================================= */

class ReplayMemory {
    constructor(capacity) {
        this.capacity = capacity;
        this.memory = [];
        this.position = 0;
    }

    add(state, action, reward, nextState, done) {
        const experience = {
            state: [...state],
            action,
            reward,
            nextState: [...nextState],
            done
        };

        if (this.memory.length < this.capacity) {
            this.memory.push(experience);
        } else {
            this.memory[this.position] = experience;
        }

        this.position = (this.position + 1) % this.capacity;
    }

    sample(size) {
        const result = [];
        const count = Math.min(size, this.memory.length);

        for (let i = 0; i < count; i++) {
            const index = Math.floor(
                Math.random() * this.memory.length
            );

            result.push(this.memory[index]);
        }

        return result;
    }

    get length() {
        return this.memory.length;
    }
}

/* =========================================================
   DQN AGENT
   ========================================================= */

class DQNAgent {
    constructor() {
        this.memory = new ReplayMemory(CONFIG.replayCapacity);

        this.epsilon = CONFIG.epsilonStart;
        this.updates = 0;
        this.experiences = 0;

        this.model = null;
        this.targetModel = null;

        this.ready = false;
    }

    createModel() {
        const model = tf.sequential();

        model.add(tf.layers.dense({
            inputShape: [5],
            units: 64,
            activation: "relu"
        }));

        model.add(tf.layers.dense({
            units: 64,
            activation: "relu"
        }));

        model.add(tf.layers.dense({
            units: 3,
            activation: "linear"
        }));

        model.compile({
            optimizer: tf.train.adam(CONFIG.learningRate),
            loss: "meanSquaredError"
        });

        return model;
    }

    async initialize() {
        this.model = this.createModel();
        this.targetModel = this.createModel();

        try {
            const loaded = await tf.loadLayersModel(
                `localstorage://${MODEL_KEY}`
            );

            this.model.dispose();
            this.model = loaded;

            this.copyWeights();
        } catch (error) {
            console.log("Starting a new AI model.");
        }

        this.ready = true;
    }

    copyWeights() {
        const weights = this.model.getWeights();
        const copies = weights.map(weight => weight.clone());

        this.targetModel.setWeights(copies);

        copies.forEach(weight => weight.dispose());
    }

    chooseAction(state) {
        if (!this.ready) return 1;

        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 3);
        }

        return tf.tidy(() => {
            const input = tf.tensor2d([state]);
            const prediction = this.model.predict(input);
            return prediction.argMax(1).dataSync()[0];
        });
    }

    remember(state, action, reward, nextState, done) {
        this.memory.add(
            state,
            action,
            reward,
            nextState,
            done
        );

        this.experiences++;
        totalExperiences++;
    }

    async learnOnce() {
        if (!this.ready || this.memory.length < CONFIG.batchSize) {
            return;
        }

        const batch = this.memory.sample(CONFIG.batchSize);

        const states = batch.map(item => item.state);
        const nextStates = batch.map(item => item.nextState);

        const actions = batch.map(item => item.action);
        const rewards = batch.map(item => item.reward);
        const dones = batch.map(item => item.done);

        const stateTensor = tf.tensor2d(states);
        const nextStateTensor = tf.tensor2d(nextStates);

        const currentQ = this.model.predict(stateTensor);
        const nextQ = this.targetModel.predict(nextStateTensor);

        const currentValues = await currentQ.array();
        const nextValues = await nextQ.array();

        currentQ.dispose();
        nextQ.dispose();

        const targets = currentValues.map((values, i) => {
            const updated = [...values];

            const futureReward = Math.max(...nextValues[i]);

            updated[actions[i]] = dones[i]
                ? rewards[i]
                : rewards[i] + CONFIG.gamma * futureReward;

            return updated;
        });

        const targetTensor = tf.tensor2d(targets);

        await this.model.fit(stateTensor, targetTensor, {
            epochs: 1,
            batchSize: CONFIG.batchSize,
            shuffle: true,
            verbose: 0
        });

        stateTensor.dispose();
        nextStateTensor.dispose();
        targetTensor.dispose();

        this.updates++;
        totalUpdates++;

        this.epsilon = Math.max(
            CONFIG.epsilonMin,
            this.epsilon * CONFIG.epsilonDecay
        );

        if (this.updates % CONFIG.targetUpdateFrequency === 0) {
            this.copyWeights();
        }

        updateGameUI();
    }

    async learnAfterPoint() {
        if (!this.ready || trainingInProgress) {
            pendingTraining = true;
            return;
        }

        trainingInProgress = true;

        try {
            for (
                let i = 0;
                i < CONFIG.trainingBatchesPerPoint;
                i++
            ) {
                if (this.memory.length < CONFIG.batchSize) {
                    break;
                }

                await this.learnOnce();
            }

            if (this.updates > 0 &&
                this.updates % CONFIG.saveFrequency === 0) {
                await this.save();
            }
        } catch (error) {
            console.error("Training error:", error);
        } finally {
            trainingInProgress = false;

            if (pendingTraining) {
                pendingTraining = false;
                this.learnAfterPoint();
            }
        }
    }

    async save() {
        if (!this.ready) return;

        try {
            await this.model.save(`localstorage://${MODEL_KEY}`);
        } catch (error) {
            console.error("Model save failed:", error);
        }
    }
}

/* =========================================================
   STATE AND PREDICTION
   ========================================================= */

function getLiveState() {
    return [
        ball.x / CONFIG.width,
        ball.y / CONFIG.height,
        ball.vx / 10,
        ball.vy / 10,
        (ai.x - ball.x) / CONFIG.width
    ];
}

function predictBallXAtY(targetY) {
    if (ball.vy <= 0) {
        return ball.x;
    }

    const time = (targetY - ball.y) / ball.vy;

    if (time <= 0) {
        return ball.x;
    }

    let predicted = ball.x + ball.vx * time;

    const left = CONFIG.ballRadius;
    const right = CONFIG.width - CONFIG.ballRadius;
    const range = right - left;

    let relative = (predicted - left) % (range * 2);

    if (relative < 0) {
        relative += range * 2;
    }

    if (relative > range) {
        relative = range * 2 - relative;
    }

    return left + relative;
}

/* =========================================================
   AI MOVEMENT
   ========================================================= */

function moveAI() {
    ai.width = getAIPaddleWidth();

    const currentState = getLiveState();

    let targetX;

    if (ball.vy < 0) {
        // When the ball moves away, AI follows slowly.
        targetX = ball.x;
    } else {
        const predictedX = predictBallXAtY(ai.y);

        // Initially the AI does not predict accurately.
        const error = getAIAimError();

        const noisyPrediction =
            predictedX + aiAimOffset * (1 - aiSkill);

        // Gradually switch from current-position tracking
        // to future-position prediction.
        const predictionWeight = 0.3 + aiSkill * 0.7;

        targetX =
            ball.x * (1 - predictionWeight) +
            noisyPrediction * predictionWeight;

        // DQN output: 0 = left, 1 = stay, 2 = right.
        if (
            agent &&
            agent.ready &&
            agent.updates >= CONFIG.learnedAIStart
        ) {
            const action = agent.chooseAction(currentState);

            lastAIAction = action;

            const learningInfluence = Math.min(
                1,
                (agent.updates - CONFIG.learnedAIStart) / 100
            );

            const adjustment =
                (action - 1) * 45 * learningInfluence;

            targetX += adjustment * aiSkill;
        }

        // Remaining uncertainty decreases as the AI improves.
        targetX +=
            (Math.random() - 0.5) * error * 0.12;
    }

    const minimum = ai.width / 2;
    const maximum = CONFIG.width - ai.width / 2;

    targetX = Math.max(
        minimum,
        Math.min(maximum, targetX)
    );

    const difference = targetX - ai.x;
    const speed = getAISpeed();

    if (Math.abs(difference) > 2) {
        ai.x += Math.sign(difference) *
            Math.min(Math.abs(difference), speed);
    }

    lastAIState = currentState;
}

/* =========================================================
   EXPERIENCE RECORDING
   ========================================================= */

function recordAIExperience(reward, done) {
    if (!agent || !agent.ready || !lastAIState) {
        return;
    }

    const nextState = getLiveState();

    // The reward is associated with the AI's latest action.
    agent.remember(
        lastAIState,
        lastAIAction,
        reward,
        nextState,
        done
    );

    lastAIState = null;
}

/* =========================================================
   BALL RESET
   ========================================================= */

function resetLiveBall() {
    ball.x = CONFIG.width / 2;
    ball.y = CONFIG.height / 2;

    const direction = Math.random() < 0.5 ? -1 : 1;

    ball.vx = (Math.random() * 2 + 1) * direction;
    ball.vy = CONFIG.ballSpeed *
        (Math.random() < 0.5 ? -1 : 1);

    // A fixed error for this serve makes the initial AI
    // visibly less accurate.
    aiAimOffset =
        (Math.random() - 0.5) * getAIAimError() * 2;

    lastAIState = null;
    lastAIAction = 1;
}

/* =========================================================
   COLLISIONS
   ========================================================= */

function ballHitsPaddle(paddle) {
    return (
        ball.x + ball.radius > paddle.x - paddle.width / 2 &&
        ball.x - ball.radius < paddle.x + paddle.width / 2 &&
        ball.y + ball.radius > paddle.y - paddle.height / 2 &&
        ball.y - ball.radius < paddle.y + paddle.height / 2
    );
}

function bounceBallFromPaddle(paddle, isAI) {
    const relative =
        (ball.x - paddle.x) / (paddle.width / 2);

    ball.vx = relative * 5;
    ball.vy = isAI
        ? Math.abs(ball.vy)
        : -Math.abs(ball.vy);

    ball.vy *= 1.02;

    if (isAI) {
        // Catching the ball is positive feedback.
        recordAIExperience(1, false);
    }

    playSound(isAI ? 520 : 650);
}

/* =========================================================
   POINT HANDLING
   ========================================================= */

async function finishPoint(playerWon) {
    if (playerWon) {
        playerScore++;
        statistics.playerPoints++;

        // AI failed to stop the player.
        recordAIExperience(-1, true);

        playSound(260, 0.15);
    } else {
        aiScore++;
        statistics.aiPoints++;

        // AI successfully scored.
        recordAIExperience(1.5, true);

        playSound(800, 0.15);
    }

    // Every completed point contributes to progression.
    advanceAISkill();

    // Train after every point, regardless of who scored.
    if (agent) {
        agent.learnAfterPoint();
    }

    updateGameUI();
    resetLiveBall();
}

/* =========================================================
   GAME PHYSICS
   ========================================================= */

function moveLive() {
    if (!running || paused) return;

    frameCounter++;

    if (keys.left) {
        player.x -= CONFIG.playerSpeed;
    }

    if (keys.right) {
        player.x += CONFIG.playerSpeed;
    }

    player.x = Math.max(
        player.width / 2,
        Math.min(CONFIG.width - player.width / 2, player.x)
    );

    moveAI();

    ball.x += ball.vx;
    ball.y += ball.vy;

    // Side walls.
    if (
        ball.x - ball.radius <= 0 ||
        ball.x + ball.radius >= CONFIG.width
    ) {
        ball.vx *= -1;
        playSound(350);
    }

    // AI paddle collision.
    if (ball.vy < 0 && ballHitsPaddle(ai)) {
        ball.y = ai.y + ai.height / 2 + ball.radius;
        bounceBallFromPaddle(ai, true);
    }

    // Player paddle collision.
    if (ball.vy > 0 && ballHitsPaddle(player)) {
        ball.y = player.y - player.height / 2 - ball.radius;
        bounceBallFromPaddle(player, false);
    }

    // Player scores.
    if (ball.y - ball.radius > CONFIG.height) {
        finishPoint(true);
    }

    // AI scores.
    if (ball.y + ball.radius < 0) {
        finishPoint(false);
    }
}

/* =========================================================
   DRAWING
   ========================================================= */

function drawArena() {
    if (!ctx) return;

    ctx.clearRect(0, 0, CONFIG.width, CONFIG.height);

    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, CONFIG.width, CONFIG.height);

    // Center line.
    ctx.strokeStyle = "#444";
    ctx.setLineDash([10, 12]);
    ctx.beginPath();
    ctx.moveTo(0, CONFIG.height / 2);
    ctx.lineTo(CONFIG.width, CONFIG.height / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // AI paddle.
    ctx.fillStyle = "#b6ff00";
    ctx.fillRect(
        ai.x - ai.width / 2,
        ai.y - ai.height / 2,
        ai.width,
        ai.height
    );

    // Player paddle.
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(
        player.x - player.width / 2,
        player.y - player.height / 2,
        player.width,
        player.height
    );

    // Ball.
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fill();

    // Score.
    ctx.fillStyle = "#b6ff00";
    ctx.font = "bold 34px monospace";
    ctx.textAlign = "center";

    ctx.fillText(
        aiScore,
        CONFIG.width / 2 - 45,
        CONFIG.height / 2 - 25
    );

    ctx.fillStyle = "#ffffff";

    ctx.fillText(
        playerScore,
        CONFIG.width / 2 + 45,
        CONFIG.height / 2 + 55
    );
}

/* =========================================================
   GAME LOOP
   ========================================================= */

function gameLoop() {
    if (!running) return;

    if (!paused) {
        moveLive();
        drawArena();
    }

    animationFrame = requestAnimationFrame(gameLoop);
}

/* =========================================================
   SCREEN MANAGEMENT
   ========================================================= */

function showScreen(name) {
    screen = name;

    if (elements.menuScreen) {
        elements.menuScreen.style.display =
            name === "menu" ? "flex" : "none";
    }

    if (elements.gameScreen) {
        elements.gameScreen.style.display =
            name === "game" ? "flex" : "none";
    }

    if (elements.statsScreen) {
        elements.statsScreen.style.display =
            name === "stats" ? "flex" : "none";
    }
}

function startMatch() {
    playerScore = 0;
    aiScore = 0;

    player.x = CONFIG.width / 2;
    ai.x = CONFIG.width / 2;

    matchStartTime = Date.now();
    matchDuration = 0;

    resetLiveBall();

    running = true;
    paused = false;

    if (elements.pauseOverlay) {
        elements.pauseOverlay.style.display = "none";
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

    if (elements.pauseOverlay) {
        elements.pauseOverlay.style.display = "flex";
    }
}

function resumeGame() {
    if (!running) return;

    paused = false;

    if (elements.pauseOverlay) {
        elements.pauseOverlay.style.display = "none";
    }
}

async function endMatch() {
    if (!running) return;

    running = false;
    paused = false;

    cancelAnimationFrame(animationFrame);

    matchDuration = Math.floor(
        (Date.now() - matchStartTime) / 1000
    );

    statistics.matches++;
    statistics.totalDuration += matchDuration;

    statistics.history.push({
        playerScore,
        aiScore,
        duration: matchDuration,
        aiLevel,
        date: new Date().toISOString()
    });

    if (statistics.history.length > 100) {
        statistics.history.shift();
    }

    saveStatistics();

    if (agent) {
        await agent.save();
    }

    showScreen("menu");
}

/* =========================================================
   STATISTICS
   ========================================================= */

function loadStatistics() {
    try {
        const saved = localStorage.getItem(STATS_KEY);

        if (saved) {
            statistics = {
                ...statistics,
                ...JSON.parse(saved)
            };
        }
    } catch (error) {
        console.warn("Could not load statistics:", error);
    }
}

function saveStatistics() {
    localStorage.setItem(
        STATS_KEY,
        JSON.stringify(statistics)
    );
}

function getWinRate() {
    const total =
        statistics.playerPoints + statistics.aiPoints;

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

function updateGameUI() {
    updateAISkill();

    if (elements.liveRate) {
        elements.liveRate.textContent =
            `${getWinRate()}%`;
    }

    if (elements.liveUpdates) {
        elements.liveUpdates.textContent =
            agent ? agent.updates : totalUpdates;
    }

    if (elements.liveSpeed) {
        elements.liveSpeed.textContent =
            `LV.${aiLevel}`;
    }

    if (elements.statMatches) {
        elements.statMatches.textContent =
            statistics.matches;
    }

    if (elements.statRate) {
        elements.statRate.textContent =
            `${getWinRate()}%`;
    }

    if (elements.statTime) {
        elements.statTime.textContent =
            formatTime(statistics.totalDuration);
    }

    if (elements.statUpdates) {
        elements.statUpdates.textContent =
            agent ? agent.updates : totalUpdates;
    }

    if (elements.statExperiences) {
        elements.statExperiences.textContent =
            agent ? agent.experiences : totalExperiences;
    }

    if (elements.statEpsilon) {
        elements.statEpsilon.textContent =
            agent ? agent.epsilon.toFixed(3) : "0.350";
    }
}

/* =========================================================
   STATISTICS CHART
   ========================================================= */

function drawStatisticsChart() {
    const chart = elements.statsChart;

    if (!chart) return;

    const chartCtx = chart.getContext("2d");

    const width = chart.width;
    const height = chart.height;

    chartCtx.clearRect(0, 0, width, height);

    chartCtx.fillStyle = "#080808";
    chartCtx.fillRect(0, 0, width, height);

    const history = statistics.history;

    if (history.length === 0) {
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

    chartCtx.strokeStyle = "#444";
    chartCtx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const y = padding + graphHeight * i / 4;

        chartCtx.beginPath();
        chartCtx.moveTo(padding, y);
        chartCtx.lineTo(width - padding, y);
        chartCtx.stroke();
    }

    const maxPoints = Math.max(
        1,
        ...history.map(item =>
            Math.max(item.playerScore, item.aiScore)
        )
    );

    function drawLine(key, color) {
        chartCtx.beginPath();
        chartCtx.strokeStyle = color;
        chartCtx.lineWidth = 2;

        history.forEach((item, index) => {
            const x = padding +
                (history.length === 1
                    ? graphWidth / 2
                    : graphWidth * index / (history.length - 1));

            const y = height - padding -
                (item[key] / maxPoints) * graphHeight;

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
    const chart = elements.statsChart;
    const tooltip = elements.chartTooltip;

    if (!chart || !tooltip) return;

    chart.addEventListener("mousemove", event => {
        const history = statistics.history;

        if (!history.length) return;

        const rect = chart.getBoundingClientRect();

        const relativeX =
            (event.clientX - rect.left) / rect.width;

        const index = Math.round(
            relativeX * (history.length - 1)
        );

        const item = history[
            Math.max(0, Math.min(history.length - 1, index))
        ];

        if (!item) return;

        tooltip.style.display = "block";
        tooltip.textContent =
            `Player: ${item.playerScore} | ` +
            `AI: ${item.aiScore} | ` +
            `AI Level: ${item.aiLevel}`;

        tooltip.style.left =
            `${event.clientX - rect.left + 10}px`;

        tooltip.style.top =
            `${event.clientY - rect.top + 10}px`;
    });

    chart.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
    });
}

function showStatistics() {
    showScreen("stats");
    updateGameUI();
    drawStatisticsChart();
}

/* =========================================================
   KEYBOARD
   ========================================================= */

document.addEventListener("keydown", event => {
    if (
        event.key === "ArrowLeft" ||
        event.key.toLowerCase() === "a"
    ) {
        keys.left = true;
    }

    if (
        event.key === "ArrowRight" ||
        event.key.toLowerCase() === "d"
    ) {
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
    if (
        event.key === "ArrowLeft" ||
        event.key.toLowerCase() === "a"
    ) {
        keys.left = false;
    }

    if (
        event.key === "ArrowRight" ||
        event.key.toLowerCase() === "d"
    ) {
        keys.right = false;
    }
});

/* =========================================================
   BUTTONS
   ========================================================= */

function setupButtons() {
    if (elements.playBtn) {
        elements.playBtn.addEventListener("click", startMatch);
    }

    if (elements.newMatchBtn) {
        elements.newMatchBtn.addEventListener("click", startMatch);
    }

    if (elements.statsBtn) {
        elements.statsBtn.addEventListener("click", showStatistics);
    }

    if (elements.statsBackBtn) {
        elements.statsBackBtn.addEventListener("click", () => {
            showScreen("menu");
        });
    }

    if (elements.gameMenuBtn) {
        elements.gameMenuBtn.addEventListener("click", endMatch);
    }

    if (elements.endMatchBtn) {
        elements.endMatchBtn.addEventListener("click", endMatch);
    }

    if (elements.pauseBtn) {
        elements.pauseBtn.addEventListener("click", pauseGame);
    }

    if (elements.resumeBtn) {
        elements.resumeBtn.addEventListener("click", resumeGame);
    }

    if (elements.resetStatsBtn) {
        elements.resetStatsBtn.addEventListener("click", () => {
            if (!confirm("Reset all statistics?")) return;

            statistics = {
                matches: 0,
                playerPoints: 0,
                aiPoints: 0,
                totalDuration: 0,
                history: []
            };

            saveStatistics();
            updateGameUI();
            drawStatisticsChart();
        });
    }
}

/* =========================================================
   INITIALIZATION
   ========================================================= */

async function init() {
    if (!canvas || !ctx) {
        console.error("Canvas element #arena was not found.");
        return;
    }

    canvas.width = CONFIG.width;
    canvas.height = CONFIG.height;

    loadStatistics();
    updateAISkill();

    setupButtons();
    setupChartHover();

    if (typeof tf === "undefined") {
        if (elements.modelStatus) {
            elements.modelStatus.textContent =
                "TensorFlow.js not loaded";
        }

        console.error("TensorFlow.js is unavailable.");
        showScreen("menu");
        return;
    }

    try {
        await tf.setBackend("cpu");
        await tf.ready();

        agent = new DQNAgent();

        await agent.initialize();

        if (elements.modelStatus) {
            elements.modelStatus.textContent =
                "AI MODEL READY";
        }
    } catch (error) {
        console.error("AI initialization failed:", error);

        if (elements.modelStatus) {
            elements.modelStatus.textContent =
                "AI INITIALIZATION ERROR";
        }
    }

    updateGameUI();
    showScreen("menu");

    drawStatisticsChart();

    console.log("Pong AI initialized.");
    console.log("AI attempts:", aiAttempts);
    console.log("AI level:", aiLevel);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
