
"use strict";

// ============================================================
// 1. CONFIGURATION
// ============================================================

const CONFIG = {
    width: 900,
    height: 620,

    paddleWidth: 125,
    paddleHeight: 13,
    ballRadius: 9,

    playerSpeed: 8,
    aiSpeed: 7.5,
    ballSpeed: 5,

    replayCapacity: 12000,
    batchSize: 32,

    gamma: 0.97,
    learningRate: 0.0005,

    epsilonStart: 0.25,
    epsilonMin: 0.03,
    epsilonDecay: 0.997,

    targetUpdateFrequency: 10,
    saveFrequency: 10,

    learnedAIStart: 10,
    maxAIAdjustment: 18
};

const MODEL_KEY = "pong-dqn-model-v4";
const $ = id => document.getElementById(id);


// ============================================================
// 2. GLOBAL STATE
// ============================================================

let screen = "menu";
let paused = false;
let running = false;

let raf = null;

let agent = null;

let saving = false;
let trainingInProgress = false;
let pendingTrainingEvents = 0;

let lastSavedUpdate = 0;
let lastStatsSave = 0;

let audioContext = null;

const keys = {
    left: false,
    right: false
};

let match = {
    player: 0,
    ai: 0,
    startTime: 0,
    duration: 0
};

let statistics = {
    matches: 0,
    playerPoints: 0,
    aiPoints: 0,
    totalDuration: 0,
    history: []
};

let liveBall = {
    x: CONFIG.width / 2,
    y: CONFIG.height / 2,
    vx: CONFIG.ballSpeed,
    vy: CONFIG.ballSpeed
};

let player = {
    x: CONFIG.width / 2,
    y: CONFIG.height - 42
};

let ai = {
    x: CONFIG.width / 2,
    y: 35
};

let lastAIState = null;
let lastAIAction = 1;


// ============================================================
// 3. AUDIO
// ============================================================

function initAudio() {
    if (!audioContext) {
        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;

        if (AudioContextClass) {
            audioContext = new AudioContextClass();
        }
    }

    if (
        audioContext &&
        audioContext.state === "suspended"
    ) {
        audioContext.resume();
    }
}

function playTone(
    frequency,
    duration,
    volume = 0.04,
    type = "square"
) {
    if (!audioContext) return;

    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = type;
    oscillator.frequency.value = frequency;

    gain.gain.setValueAtTime(
        volume,
        audioContext.currentTime
    );

    gain.gain.exponentialRampToValueAtTime(
        0.001,
        audioContext.currentTime + duration
    );

    oscillator.connect(gain);
    gain.connect(audioContext.destination);

    oscillator.start();
    oscillator.stop(audioContext.currentTime + duration);
}

function playButtonSound() {
    playTone(650, 0.06, 0.025);
}

function playWallSound() {
    playTone(260, 0.045, 0.025);
}

function playPaddleSound() {
    playTone(420, 0.07, 0.04);
}

function playScoreSound() {
    playTone(520, 0.12, 0.05);

    setTimeout(() => {
        playTone(780, 0.15, 0.04);
    }, 100);
}

function playStartSound() {
    playTone(440, 0.1);

    setTimeout(() => {
        playTone(660, 0.1);
    }, 100);
}

function playEndSound() {
    playTone(300, 0.2);

    setTimeout(() => {
        playTone(200, 0.25);
    }, 180);
}


// ============================================================
// 4. REPLAY MEMORY
// ============================================================

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

        this.position =
            (this.position + 1) % this.capacity;
    }

    sample(batchSize) {
        const result = [];

        for (let i = 0; i < batchSize; i++) {
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


// ============================================================
// 5. DQN AGENT
// ============================================================

class DQNAgent {
    constructor() {
        this.memory = new ReplayMemory(
            CONFIG.replayCapacity
        );

        this.epsilon = CONFIG.epsilonStart;
        this.updates = 0;
        this.experiences = 0;

        this.model = null;
        this.target = null;

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
        try {
            this.model = await tf.loadLayersModel(
                `localstorage://${MODEL_KEY}`
            );

            console.log("Previous AI model loaded.");

        } catch (error) {
            console.log("Creating a new AI model.");

            this.model = this.createModel();
        }

        this.target = this.createModel();
        this.copyWeights();

        const epsilon = Number(
            localStorage.getItem("pong-ai-epsilon")
        );

        const updates = Number(
            localStorage.getItem("pong-ai-updates")
        );

        const experiences = Number(
            localStorage.getItem("pong-ai-experiences")
        );

        if (Number.isFinite(epsilon) && epsilon > 0) {
            this.epsilon = epsilon;
        }

        this.updates = Number.isFinite(updates) ? updates : 0;
        this.experiences = Number.isFinite(experiences)
            ? experiences
            : 0;

        this.ready = true;

        updateModelStatus("AI MODEL READY");
        updateGameUI();
    }

    copyWeights() {
        const weights = this.model.getWeights().map(
            weight => weight.clone()
        );

        try {
            this.target.setWeights(weights);
        } finally {
            weights.forEach(weight => weight.dispose());
        }
    }

    act(state, training = false) {
        if (!this.ready || !this.model) return 1;

        if (
            training &&
            Math.random() < this.epsilon
        ) {
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
    }

    async learnOnce() {
        if (
            !this.ready ||
            this.memory.length < CONFIG.batchSize
        ) {
            return false;
        }

        const batch = this.memory.sample(CONFIG.batchSize);

        const states = batch.map(item => item.state);
        const nextStates = batch.map(item => item.nextState);

        const actions = batch.map(item => item.action);
        const rewards = batch.map(item => item.reward);
        const dones = batch.map(item => item.done);

        let stateTensor;
        let nextTensor;
        let currentQ;
        let nextQ;
        let targetTensor;

        try {
            stateTensor = tf.tensor2d(states);
            nextTensor = tf.tensor2d(nextStates);

            currentQ = this.model.predict(stateTensor);
            nextQ = this.target.predict(nextTensor);

            const currentValues = await currentQ.array();
            const nextValues = await nextQ.array();

            const targets = currentValues.map((values, i) => {
                const nextMax = Math.max(...nextValues[i]);

                const targetValue = dones[i]
                    ? rewards[i]
                    : rewards[i] + CONFIG.gamma * nextMax;

                const updated = [...values];
                updated[actions[i]] = targetValue;

                return updated;
            });

            targetTensor = tf.tensor2d(targets);

            await this.model.fit(
                stateTensor,
                targetTensor,
                {
                    epochs: 1,
                    batchSize: CONFIG.batchSize,
                    verbose: 0
                }
            );

            this.updates++;

            this.epsilon = Math.max(
                CONFIG.epsilonMin,
                this.epsilon * CONFIG.epsilonDecay
            );

            if (
                this.updates % CONFIG.targetUpdateFrequency === 0
            ) {
                this.copyWeights();
            }

            return true;

        } finally {
            tf.dispose([
                stateTensor,
                nextTensor,
                currentQ,
                nextQ,
                targetTensor
            ].filter(Boolean));
        }
    }

    async save() {
        if (!this.ready || saving) return;

        saving = true;

        try {
            await this.model.save(
                `localstorage://${MODEL_KEY}`
            );

            localStorage.setItem(
                "pong-ai-epsilon",
                String(this.epsilon)
            );

            localStorage.setItem(
                "pong-ai-updates",
                String(this.updates)
            );

            localStorage.setItem(
                "pong-ai-experiences",
                String(this.experiences)
            );

            lastSavedUpdate = this.updates;

            console.log("AI model saved:", this.updates);

        } catch (error) {
            console.error("Model save failed:", error);

        } finally {
            saving = false;
        }
    }
}


// ============================================================
// 6. AI PREDICTION
// ============================================================

function getLiveState() {
    return [
        liveBall.x / CONFIG.width,
        liveBall.y / CONFIG.height,
        liveBall.vx / 10,
        liveBall.vy / 10,
        (liveBall.x - ai.x) / CONFIG.width
    ];
}

function predictBallXAtY(targetY) {
    const minX = CONFIG.ballRadius;
    const maxX = CONFIG.width - CONFIG.ballRadius;
    const range = maxX - minX;

    if (liveBall.vy >= 0) {
        return CONFIG.width / 2;
    }

    const time =
        (targetY - liveBall.y) / liveBall.vy;

    if (time <= 0) {
        return liveBall.x;
    }

    const projectedX =
        liveBall.x + liveBall.vx * time;

    const period = range * 2;

    let position =
        (projectedX - minX) % period;

    if (position < 0) {
        position += period;
    }

    if (position > range) {
        position = period - position;
    }

    return minX + position;
}

function moveAI() {
    const state = getLiveState();

    const targetY =
        ai.y +
        CONFIG.paddleHeight / 2 +
        CONFIG.ballRadius;

    let targetX = predictBallXAtY(targetY);

    // Basic predictive controller.
    const difference = targetX - ai.x;

    let action = 1;

    if (difference < -5) action = 0;
    if (difference > 5) action = 2;

    // DQN gradually adjusts the prediction.
    if (
        agent &&
        agent.ready &&
        agent.updates >= CONFIG.learnedAIStart
    ) {
        const learnedAction = agent.act(state, false);

        targetX +=
            (learnedAction - 1) *
            CONFIG.maxAIAdjustment;
    }

    lastAIState = [...state];
    lastAIAction = action;

    const movement = Math.max(
        -CONFIG.aiSpeed,
        Math.min(CONFIG.aiSpeed, targetX - ai.x)
    );

    ai.x += movement;

    ai.x = Math.max(
        CONFIG.paddleWidth / 2,
        Math.min(
            CONFIG.width - CONFIG.paddleWidth / 2,
            ai.x
        )
    );
}


// ============================================================
// 7. REAL GAME EXPERIENCE
// ============================================================

function recordAIExperience(reward, done = false) {
    if (
        !agent ||
        !agent.ready ||
        !lastAIState
    ) {
        return;
    }

    const nextState = getLiveState();

    agent.remember(
        lastAIState,
        lastAIAction,
        reward,
        nextState,
        done
    );

    lastAIState = null;
}


// ============================================================
// 8. TRAIN AFTER SCORING
// ============================================================

async function trainAfterPoint() {
    if (!agent || !agent.ready) return;

    pendingTrainingEvents++;

    if (trainingInProgress) return;

    trainingInProgress = true;

    try {
        while (pendingTrainingEvents > 0) {
            pendingTrainingEvents--;

            await agent.learnOnce();

            updateGameUI();

            if (
                agent.updates > 0 &&
                agent.updates >=
                    lastSavedUpdate + CONFIG.saveFrequency
            ) {
                await agent.save();
            }
        }

    } catch (error) {
        console.error("Training failed:", error);

    } finally {
        trainingInProgress = false;
        updateGameUI();
    }
}


// ============================================================
// 9. BALL AND PLAYER MOVEMENT
// ============================================================

function resetLiveBall() {
    liveBall.x = CONFIG.width / 2;
    liveBall.y = CONFIG.height / 2;

    const horizontalDirection =
        Math.random() < 0.5 ? -1 : 1;

    const verticalDirection =
        Math.random() < 0.5 ? -1 : 1;

    liveBall.vx =
        horizontalDirection * CONFIG.ballSpeed * 0.6;

    liveBall.vy =
        verticalDirection * CONFIG.ballSpeed;

    lastAIState = null;
}

function movePlayer() {
    if (keys.left) {
        player.x -= CONFIG.playerSpeed;
    }

    if (keys.right) {
        player.x += CONFIG.playerSpeed;
    }

    player.x = Math.max(
        CONFIG.paddleWidth / 2,
        Math.min(
            CONFIG.width - CONFIG.paddleWidth / 2,
            player.x
        )
    );
}


// ============================================================
// 10. LIVE GAME PHYSICS
// ============================================================

function moveLive() {
    moveAI();
    movePlayer();

    liveBall.x += liveBall.vx;
    liveBall.y += liveBall.vy;

    // Side walls.
    if (
        liveBall.x - CONFIG.ballRadius <= 0 ||
        liveBall.x + CONFIG.ballRadius >= CONFIG.width
    ) {
        liveBall.vx *= -1;
        playWallSound();
    }

    // AI paddle.
    const aiCollision =
        liveBall.vy < 0 &&
        liveBall.y - CONFIG.ballRadius <=
            ai.y + CONFIG.paddleHeight / 2 &&
        liveBall.y + CONFIG.ballRadius >=
            ai.y - CONFIG.paddleHeight / 2 &&
        liveBall.x >=
            ai.x - CONFIG.paddleWidth / 2 &&
        liveBall.x <=
            ai.x + CONFIG.paddleWidth / 2;

    if (aiCollision) {
        liveBall.vy = Math.abs(liveBall.vy);

        const offset =
            (liveBall.x - ai.x) /
            (CONFIG.paddleWidth / 2);

        liveBall.vx = offset * 5;

        recordAIExperience(1, false);

        playPaddleSound();
    }

    // Player paddle.
    const playerCollision =
        liveBall.vy > 0 &&
        liveBall.y + CONFIG.ballRadius >=
            player.y - CONFIG.paddleHeight / 2 &&
        liveBall.y - CONFIG.ballRadius <=
            player.y + CONFIG.paddleHeight / 2 &&
        liveBall.x >=
            player.x - CONFIG.paddleWidth / 2 &&
        liveBall.x <=
            player.x + CONFIG.paddleWidth / 2;

    if (playerCollision) {
        liveBall.vy = -Math.abs(liveBall.vy);

        const offset =
            (liveBall.x - player.x) /
            (CONFIG.paddleWidth / 2);

        liveBall.vx = offset * 5;

        playPaddleSound();
    }

    // Player scores.
    if (liveBall.y < -CONFIG.ballRadius) {
        match.player++;

        recordAIExperience(-1, true);

        playScoreSound();
        resetLiveBall();

        trainAfterPoint();
        updateGameUI();
    }

    // AI scores.
    if (liveBall.y > CONFIG.height + CONFIG.ballRadius) {
        match.ai++;

        playScoreSound();
        resetLiveBall();

        trainAfterPoint();
        updateGameUI();
    }
}


// ============================================================
// 11. DRAW GAME
// ============================================================

function drawArena() {
    const canvas = $("arena");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, CONFIG.width, CONFIG.height);

    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, CONFIG.width, CONFIG.height);

    // Center line.
    ctx.strokeStyle = "#333333";
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 12]);

    ctx.beginPath();
    ctx.moveTo(0, CONFIG.height / 2);
    ctx.lineTo(CONFIG.width, CONFIG.height / 2);
    ctx.stroke();

    ctx.setLineDash([]);

    // AI paddle.
    ctx.fillStyle = "#c7ff4d";

    ctx.fillRect(
        ai.x - CONFIG.paddleWidth / 2,
        ai.y - CONFIG.paddleHeight / 2,
        CONFIG.paddleWidth,
        CONFIG.paddleHeight
    );

    // Player paddle.
    ctx.fillStyle = "#ffffff";

    ctx.fillRect(
        player.x - CONFIG.paddleWidth / 2,
        player.y - CONFIG.paddleHeight / 2,
        CONFIG.paddleWidth,
        CONFIG.paddleHeight
    );

    // Ball.
    ctx.beginPath();

    ctx.arc(
        liveBall.x,
        liveBall.y,
        CONFIG.ballRadius,
        0,
        Math.PI * 2
    );

    ctx.fillStyle = "#c7ff4d";
    ctx.shadowColor = "#c7ff4d";
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Scores.
    ctx.textAlign = "center";
    ctx.font = "bold 32px monospace";

    ctx.fillStyle = "#c7ff4d";
    ctx.fillText(match.ai, CONFIG.width / 2, 100);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(
        match.player,
        CONFIG.width / 2,
        CONFIG.height - 85
    );

    ctx.font = "13px monospace";
    ctx.fillStyle = "#888888";

    ctx.fillText("AI", 55, 40);
    ctx.fillText("PLAYER", 65, CONFIG.height - 25);
}


// ============================================================
// 12. GAME LOOP
// ============================================================

function gameLoop(now) {
    if (screen !== "game") return;

    if (!paused) {
        moveLive();

        if (now - lastStatsSave > 5000) {
            saveStats();
            lastStatsSave = now;
        }
    }

    drawArena();

    raf = requestAnimationFrame(gameLoop);
}


// ============================================================
// 13. SCREEN MANAGEMENT
// ============================================================

function showScreen(name) {
    const screens = [
        "menuScreen",
        "gameScreen",
        "statsScreen"
    ];

    screens.forEach(id => {
        $(id)?.classList.add("hidden");
    });

    const target = {
        menu: "menuScreen",
        game: "gameScreen",
        stats: "statsScreen"
    }[name];

    if (target) {
        $(target)?.classList.remove("hidden");
    }
}

function showPauseOverlay() {
    $("pauseOverlay")?.classList.remove("hidden");
}

function hidePauseOverlay() {
    $("pauseOverlay")?.classList.add("hidden");
}


// ============================================================
// 14. GAME CONTROL
// ============================================================

function startMatch() {
    if (!agent || !agent.ready) {
        updateModelStatus("AI IS STILL LOADING");
        return;
    }

    screen = "game";
    paused = false;
    running = true;

    match = {
        player: 0,
        ai: 0,
        startTime: Date.now(),
        duration: 0
    };

    player.x = CONFIG.width / 2;
    ai.x = CONFIG.width / 2;

    resetLiveBall();

    showScreen("game");
    hidePauseOverlay();

    playStartSound();

    if (raf) {
        cancelAnimationFrame(raf);
    }

    raf = requestAnimationFrame(gameLoop);

    updateGameUI();
}

function pauseGame() {
    if (screen !== "game") return;

    paused = !paused;

    if (paused) {
        showPauseOverlay();
    } else {
        hidePauseOverlay();
    }

    updateGameUI();
}

function endMatch() {
    if (!running) return;

    running = false;
    paused = false;

    if (raf) {
        cancelAnimationFrame(raf);
        raf = null;
    }

    match.duration = Math.floor(
        (Date.now() - match.startTime) / 1000
    );

    statistics.matches++;
    statistics.playerPoints += match.player;
    statistics.aiPoints += match.ai;
    statistics.totalDuration += match.duration;

    const totalPoints = match.player + match.ai;

    const aiRate = totalPoints > 0
        ? match.ai / totalPoints * 100
        : 0;

    const playerRate = totalPoints > 0
        ? match.player / totalPoints * 100
        : 0;

    statistics.history.push({
        player: match.player,
        ai: match.ai,
        playerRate,
        aiRate,
        duration: match.duration,
        updates: agent ? agent.updates : 0
    });

    if (statistics.history.length > 30) {
        statistics.history.shift();
    }

    saveStats();

    if (agent) {
        agent.save();
    }

    playEndSound();

    screen = "menu";
    showScreen("menu");
}


// ============================================================
// 15. STATISTICS
// ============================================================

function getAIRate() {
    const total =
        statistics.playerPoints + statistics.aiPoints;

    if (total === 0) return 0;

    return statistics.aiPoints / total * 100;
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;

    return `${minutes}m ${remaining}s`;
}

function saveStats() {
    try {
        localStorage.setItem(
            "pong-statistics",
            JSON.stringify(statistics)
        );
    } catch (error) {
        console.error("Statistics save failed:", error);
    }
}

function loadStats() {
    try {
        const saved = localStorage.getItem("pong-statistics");

        if (saved) {
            const parsed = JSON.parse(saved);

            statistics = {
                ...statistics,
                ...parsed
            };

            if (!Array.isArray(statistics.history)) {
                statistics.history = [];
            }
        }
    } catch (error) {
        console.error("Statistics load failed:", error);
    }
}

function updateGameUI() {
    const rate = getAIRate();

    if ($("liveRate")) {
        $("liveRate").textContent =
            `${rate.toFixed(1)}%`;
    }

    if ($("liveUpdates")) {
        $("liveUpdates").textContent =
            agent ? agent.updates : 0;
    }

    if ($("liveSpeed")) {
        $("liveSpeed").textContent = "NORMAL";
    }

    if ($("statMatches")) {
        $("statMatches").textContent = statistics.matches;
    }

    if ($("statRate")) {
        $("statRate").textContent =
            `${rate.toFixed(1)}%`;
    }

    if ($("statTime")) {
        $("statTime").textContent =
            formatTime(statistics.totalDuration);
    }

    if ($("statUpdates")) {
        $("statUpdates").textContent =
            agent ? agent.updates : 0;
    }

    if ($("statExperiences")) {
        $("statExperiences").textContent =
            agent ? agent.experiences : 0;
    }

    if ($("statEpsilon")) {
        $("statEpsilon").textContent =
            agent ? agent.epsilon.toFixed(3) : "0.250";
    }

    drawStatsChart();
}


// ============================================================
// 16. STATISTICS GRAPH
// ============================================================

function getChartRate(item, key, scoreKey) {
    if (Number.isFinite(item[key])) {
        return item[key];
    }

    const total =
        (item.player || 0) + (item.ai || 0);

    return total > 0
        ? (item[scoreKey] || 0) / total * 100
        : 0;
}

function drawStatsChart() {
    const canvas = $("statsChart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#080808";
    ctx.fillRect(0, 0, width, height);

    const history = statistics.history || [];

    canvas.chartSeries = [];

    if (history.length === 0) {
        ctx.fillStyle = "#888888";
        ctx.font = "14px monospace";
        ctx.textAlign = "center";

        ctx.fillText(
            "NO MATCH DATA",
            width / 2,
            height / 2
        );

        return;
    }

    const padding = {
        left: 55,
        right: 25,
        top: 25,
        bottom: 40
    };

    const graphWidth =
        width - padding.left - padding.right;

    const graphHeight =
        height - padding.top - padding.bottom;

    // Percentage grid.
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.fillStyle = "#888888";

    for (
        let percentage = 0;
        percentage <= 100;
        percentage += 20
    ) {
        const y = padding.top +
            graphHeight * (1 - percentage / 100);

        ctx.strokeStyle = "#252525";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();

        ctx.fillText(
            `${percentage}%`,
            padding.left - 10,
            y + 4
        );
    }

    const series = [
        {
            key: "playerRate",
            scoreKey: "player",
            name: "PLAYER",
            color: "#ffffff"
        },
        {
            key: "aiRate",
            scoreKey: "ai",
            name: "AI",
            color: "#c7ff4d"
        }
    ];

    series.forEach(item => {
        item.points = history.map((matchItem, index) => {
            const rate = getChartRate(
                matchItem,
                item.key,
                item.scoreKey
            );

            const x = history.length === 1
                ? padding.left + graphWidth / 2
                : padding.left +
                    index / (history.length - 1) *
                    graphWidth;

            const y = padding.top +
                graphHeight * (1 - rate / 100);

            return { x, y, rate };
        });

        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";

        ctx.beginPath();

        item.points.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });

        ctx.stroke();

        item.points.forEach(point => {
            ctx.fillStyle = item.color;

            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
            ctx.fill();
        });
    });

    canvas.chartSeries = series;

    ctx.fillStyle = "#888888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";

    ctx.fillText(
        "MATCH PROGRESSION",
        width / 2,
        height - 8
    );
}


// ============================================================
// 17. CHART HOVER
// ============================================================

function distanceToSegment(px, py, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;

    const lengthSquared = dx * dx + dy * dy;

    if (lengthSquared === 0) {
        return Math.hypot(px - a.x, py - a.y);
    }

    const t = Math.max(
        0,
        Math.min(
            1,
            ((px - a.x) * dx + (py - a.y) * dy) /
                lengthSquared
        )
    );

    const nearestX = a.x + t * dx;
    const nearestY = a.y + t * dy;

    return Math.hypot(
        px - nearestX,
        py - nearestY
    );
}

function setupChartHover() {
    const canvas = $("statsChart");
    const tooltip = $("chartTooltip");

    if (!canvas || !tooltip) return;

    canvas.addEventListener("mousemove", event => {
        const rect = canvas.getBoundingClientRect();

        const mouseX =
            (event.clientX - rect.left) *
            canvas.width / rect.width;

        const mouseY =
            (event.clientY - rect.top) *
            canvas.height / rect.height;

        const seriesList = canvas.chartSeries || [];

        let nearest = null;
        let minimumDistance = 12;

        seriesList.forEach(series => {
            const points = series.points;

            if (!points || points.length === 0) return;

            if (points.length === 1) {
                const p = points[0];

                const distance = Math.hypot(
                    mouseX - p.x,
                    mouseY - p.y
                );

                if (distance < minimumDistance) {
                    minimumDistance = distance;

                    nearest = {
                        series,
                        index: 0
                    };
                }

                return;
            }

            for (let i = 0; i < points.length - 1; i++) {
                const distance = distanceToSegment(
                    mouseX,
                    mouseY,
                    points[i],
                    points[i + 1]
                );

                if (distance < minimumDistance) {
                    minimumDistance = distance;

                    const d1 = Math.hypot(
                        mouseX - points[i].x,
                        mouseY - points[i].y
                    );

                    const d2 = Math.hypot(
                        mouseX - points[i + 1].x,
                        mouseY - points[i + 1].y
                    );

                    nearest = {
                        series,
                        index: d1 <= d2 ? i : i + 1
                    };
                }
            }
        });

        if (!nearest) {
            tooltip.classList.add("hidden");
            canvas.style.cursor = "default";
            return;
        }

        const { series, index } = nearest;
        const item = statistics.history[index];

        const rate = getChartRate(
            item,
            series.key,
            series.scoreKey
        );

        const score = series.scoreKey === "ai"
            ? item.ai
            : item.player;

        tooltip.innerHTML = `
            <div class="tooltip-title">
                ${series.name}
            </div>

            <div>Match: ${index + 1}</div>

            <div class="${series.name === "AI"
                ? "tooltip-ai"
                : "tooltip-player"}">
                Score: ${score}
            </div>

            <div style="color:${series.color}">
                Rate: ${rate.toFixed(1)}%
            </div>

            <div style="color:#888888">
                Total points: ${(item.ai || 0) + (item.player || 0)}
            </div>
        `;

        tooltip.classList.remove("hidden");

        const container =
            canvas.closest(".chart-container");

        const containerRect =
            container.getBoundingClientRect();

        const tooltipWidth = tooltip.offsetWidth;
        const tooltipHeight = tooltip.offsetHeight;

        let left =
            event.clientX - containerRect.left + 15;

        let top =
            event.clientY - containerRect.top + 15;

        left = Math.min(
            left,
            container.clientWidth - tooltipWidth - 8
        );

        top = Math.min(
            top,
            container.clientHeight - tooltipHeight - 8
        );

        tooltip.style.left =
            `${Math.max(8, left)}px`;

        tooltip.style.top =
            `${Math.max(8, top)}px`;

        canvas.style.cursor = "crosshair";
    });

    canvas.addEventListener("mouseleave", () => {
        tooltip.classList.add("hidden");
        canvas.style.cursor = "default";
    });
}


// ============================================================
// 18. MODEL STATUS
// ============================================================

function updateModelStatus(message) {
    const element = $("modelStatus");

    if (element) {
        element.textContent = message;
    }
}


// ============================================================
// 19. INPUT
// ============================================================

function setupInput() {
    document.addEventListener("keydown", event => {
        const key = event.key.toLowerCase();

        if (
            event.key === "ArrowLeft" ||
            key === "a"
        ) {
            keys.left = true;
        }

        if (
            event.key === "ArrowRight" ||
            key === "d"
        ) {
            keys.right = true;
        }

        if (event.code === "Space") {
            event.preventDefault();

            if (screen === "game") {
                pauseGame();
            }
        }
    });

    document.addEventListener("keyup", event => {
        const key = event.key.toLowerCase();

        if (
            event.key === "ArrowLeft" ||
            key === "a"
        ) {
            keys.left = false;
        }

        if (
            event.key === "ArrowRight" ||
            key === "d"
        ) {
            keys.right = false;
        }
    });

    window.addEventListener("blur", () => {
        keys.left = false;
        keys.right = false;
    });
}


// ============================================================
// 20. BUTTONS
// ============================================================

function setupButtons() {
    $("playBtn")?.addEventListener("click", () => {
        initAudio();
        startMatch();
    });

    $("statsBtn")?.addEventListener("click", () => {
        initAudio();

        screen = "stats";

        showScreen("stats");
        updateGameUI();
    });

    $("statsBackBtn")?.addEventListener("click", () => {
        screen = "menu";
        showScreen("menu");
    });

    $("gameMenuBtn")?.addEventListener("click", () => {
        endMatch();
    });

    $("endMatchBtn")?.addEventListener("click", () => {
        endMatch();
    });

    $("newMatchBtn")?.addEventListener("click", () => {
        startMatch();
    });

    $("pauseBtn")?.addEventListener("click", () => {
        pauseGame();
    });

    $("resumeBtn")?.addEventListener("click", () => {
        if (paused) {
            pauseGame();
        }
    });

    $("resetStatsBtn")?.addEventListener("click", () => {
        statistics = {
            matches: 0,
            playerPoints: 0,
            aiPoints: 0,
            totalDuration: 0,
            history: []
        };

        saveStats();
        updateGameUI();
    });

    document.addEventListener("click", event => {
        const button = event.target.closest("button");

        if (button) {
            initAudio();
            playButtonSound();
        }
    }, true);
}


// ============================================================
// 21. INITIALIZATION
// ============================================================

async function init() {
    try {
        updateModelStatus("INITIALIZING AI...");

        await tf.setBackend("cpu");
        await tf.ready();

        console.log(
            "TensorFlow backend:",
            tf.getBackend()
        );

        loadStats();

        agent = new DQNAgent();

        await agent.initialize();

        setupInput();
        setupButtons();
        setupChartHover();

        showScreen("menu");

        updateModelStatus("AI READY");
        updateGameUI();

        console.log("Pong AI initialized.");

    } catch (error) {
        console.error("Initialization failed:", error);

        updateModelStatus("AI INITIALIZATION FAILED");
    }
}

document.addEventListener("DOMContentLoaded", init);
