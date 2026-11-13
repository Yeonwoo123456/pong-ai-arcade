
"use strict";

// ============================================================
// 1. CONFIGURATION
// ============================================================

const CONFIG = {
    width: 900,
    height: 620,

    paddleWidth: 125,
    paddleHeight: 13,
    playerSpeed: 8,

    ballRadius: 9,
    ballSpeed: 5,

    aiSpeed: 6.5,

    parallelEnvironments: 32,

    replayCapacity: 12000,
    batchSize: 64,

    gamma: 0.97,
    learningRate: 0.0005,

    epsilonStart: 1.0,
    epsilonMin: 0.05,
    epsilonDecay: 0.995,

    targetUpdateFrequency: 10,

    saveFrequency: 10
};

const MODEL_KEY = "pong-dqn-model-v3";

const $ = id => document.getElementById(id);

// ============================================================
// 2. GLOBAL STATE
// ============================================================

let screen = "menu";
let paused = false;
let running = false;

let raf = null;
let lastFrameTime = 0;

let agent = null;
let envs = null;

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
    updates: 0,
    experiences: 0,
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

// ============================================================
// 3. AUDIO SYSTEM
// ============================================================

function initAudio() {
    if (!audioContext) {
        const AudioContextClass =
            window.AudioContext || window.webkitAudioContext;

        if (AudioContextClass) {
            audioContext = new AudioContextClass();
        }
    }

    if (audioContext && audioContext.state === "suspended") {
        audioContext.resume();
    }
}

function playTone(frequency, duration, volume = 0.05, type = "square") {
    if (!audioContext) return;

    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = type;
    oscillator.frequency.value = frequency;

    gain.gain.setValueAtTime(volume, audioContext.currentTime);
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
    setTimeout(() => playTone(780, 0.15, 0.04), 100);
}

function playStartSound() {
    playTone(440, 0.1);
    setTimeout(() => playTone(660, 0.1), 100);
}

function playEndSound() {
    playTone(300, 0.2);
    setTimeout(() => playTone(200, 0.25), 180);
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

        this.position = (this.position + 1) % this.capacity;
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
        this.memory = new ReplayMemory(CONFIG.replayCapacity);

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
        this.model = this.createModel();
        this.target = this.createModel();

        try {
            await this.model.loadWeights(
                `localstorage://${MODEL_KEY}`
            );
        } catch (error) {
            console.log("No previous model found. Starting fresh.");
        }

        this.copyWeights();

        this.epsilon = Number(
            localStorage.getItem("pong-ai-epsilon")
        ) || CONFIG.epsilonStart;

        this.updates = Number(
            localStorage.getItem("pong-ai-updates")
        ) || 0;

        this.experiences = Number(
            localStorage.getItem("pong-ai-experiences")
        ) || 0;

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

        if (training && Math.random() < this.epsilon) {
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

        const stateTensor = tf.tensor2d(states);
        const nextTensor = tf.tensor2d(nextStates);

        const currentQ = this.model.predict(stateTensor);
        const nextQ = this.target.predict(nextTensor);

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

        const targetTensor = tf.tensor2d(targets);

        await this.model.fit(
            stateTensor,
            targetTensor,
            {
                epochs: 1,
                batchSize: CONFIG.batchSize,
                verbose: 0
            }
        );

        tf.dispose([
            stateTensor,
            nextTensor,
            currentQ,
            nextQ,
            targetTensor
        ]);

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

            console.log(
                "AI model saved.",
                "Updates:",
                this.updates
            );

        } catch (error) {
            console.error("Model save failed:", error);

        } finally {
            saving = false;
        }
    }
}

// ============================================================
// 6. TRAINING ENVIRONMENTS
// ============================================================

class TrainingEnvironments {
    constructor(count) {
        this.count = count;

        this.worlds = Array.from(
            { length: count },
            () => this.createWorld()
        );
    }

    createWorld() {
        return {
            ballX: CONFIG.width / 2,
            ballY: CONFIG.height / 2,

            vx: (Math.random() < 0.5 ? -1 : 1) *
                CONFIG.ballSpeed,

            vy: CONFIG.ballSpeed,

            aiX: CONFIG.width / 2,
            playerX: CONFIG.width / 2,

            done: false
        };
    }

    state(index) {
        const world = this.worlds[index];

        return [
            world.ballX / CONFIG.width,
            world.ballY / CONFIG.height,
            world.vx / 10,
            world.vy / 10,
            (world.ballX - world.aiX) / CONFIG.width
        ];
    }

    step(actions) {
        const transitions = [];

        for (let i = 0; i < this.count; i++) {
            const world = this.worlds[i];

            const oldState = this.state(i);

            const action = actions[i];

            // 0 = left, 1 = stay, 2 = right
            if (action === 0) world.aiX -= CONFIG.aiSpeed;
            if (action === 2) world.aiX += CONFIG.aiSpeed;

            world.aiX = Math.max(
                CONFIG.paddleWidth / 2,
                Math.min(
                    CONFIG.width - CONFIG.paddleWidth / 2,
                    world.aiX
                )
            );

            world.ballX += world.vx;
            world.ballY += world.vy;

            let reward = 0;
            let done = false;

            if (
                world.ballX <= CONFIG.ballRadius ||
                world.ballX >= CONFIG.width - CONFIG.ballRadius
            ) {
                world.vx *= -1;
            }

            const aiHit =
                world.vy < 0 &&
                world.ballY - CONFIG.ballRadius <=
                    CONFIG.paddleHeight + 42 &&
                world.ballX >=
                    world.aiX - CONFIG.paddleWidth / 2 &&
                world.ballX <=
                    world.aiX + CONFIG.paddleWidth / 2;

            if (aiHit) {
                world.vy = Math.abs(world.vy);
                reward = 1;
            }

            if (world.ballY > CONFIG.height) {
                reward = -1;
                done = true;
            }

            if (world.ballY < 0) {
                reward = 1;
                done = true;
            }

            const nextState = this.state(i);

            transitions.push([
                oldState,
                action,
                reward,
                nextState,
                done
            ]);

            if (done) {
                this.worlds[i] = this.createWorld();
            }
        }

        return transitions;
    }
}

// ============================================================
// 7. POINT-BASED TRAINING
// ============================================================

async function trainAfterPoint() {
    if (!agent || !envs || !agent.ready) return;

    // Queue scoring events instead of dropping them.
    pendingTrainingEvents++;

    if (trainingInProgress) return;

    trainingInProgress = true;

    try {
        while (pendingTrainingEvents > 0) {
            pendingTrainingEvents--;

            // Collect one batch of experiences from parallel worlds.
            const states = envs.worlds.map(
                (_, index) => envs.state(index)
            );

            const actions = states.map(
                state => agent.act(state, true)
            );

            const transitions = envs.step(actions);

            transitions.forEach(transition => {
                agent.remember(...transition);
            });

            // Exactly one DQN update for this scoring event.
            await agent.learnOnce();

            updateGameUI();

            if (
                agent.updates > 0 &&
                agent.updates >= lastSavedUpdate + CONFIG.saveFrequency
            ) {
                await agent.save();
            }
        }

    } catch (error) {
        console.error("Point training failed:", error);

    } finally {
        trainingInProgress = false;
        updateGameUI();
    }
}

// ============================================================
// 8. LIVE GAME
// ============================================================

function resetLiveBall() {
    liveBall.x = CONFIG.width / 2;
    liveBall.y = CONFIG.height / 2;

    const direction = Math.random() < 0.5 ? -1 : 1;

    liveBall.vx = CONFIG.ballSpeed *
        (Math.random() < 0.5 ? -0.6 : 0.6);

    liveBall.vy = CONFIG.ballSpeed * direction;
}

function getLiveState() {
    return [
        liveBall.x / CONFIG.width,
        liveBall.y / CONFIG.height,
        liveBall.vx / 10,
        liveBall.vy / 10,
        (liveBall.x - ai.x) / CONFIG.width
    ];
}

function moveAI() {
    if (!agent || !agent.ready) {
        const difference = liveBall.x - ai.x;

        ai.x += Math.sign(difference) *
            Math.min(Math.abs(difference), CONFIG.aiSpeed);

        return;
    }

    // Inference only. No training or weight updates here.
    const action = agent.act(getLiveState(), false);

    if (action === 0) ai.x -= CONFIG.aiSpeed;
    if (action === 2) ai.x += CONFIG.aiSpeed;

    ai.x = Math.max(
        CONFIG.paddleWidth / 2,
        Math.min(
            CONFIG.width - CONFIG.paddleWidth / 2,
            ai.x
        )
    );
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

function moveLive() {
    moveAI();
    movePlayer();

    liveBall.x += liveBall.vx;
    liveBall.y += liveBall.vy;

    // Side walls
    if (
        liveBall.x - CONFIG.ballRadius <= 0 ||
        liveBall.x + CONFIG.ballRadius >= CONFIG.width
    ) {
        liveBall.vx *= -1;
        playWallSound();
    }

    // AI paddle collision
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

        playPaddleSound();
    }

    // Player paddle collision
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

    // Player scores when AI misses.
    if (liveBall.y < -CONFIG.ballRadius) {
        match.player++;

        playScoreSound();
        resetLiveBall();

        // Training happens only after a point.
        trainAfterPoint();

        updateGameUI();
    }

    // AI scores when player misses.
    if (liveBall.y > CONFIG.height + CONFIG.ballRadius) {
        match.ai++;

        playScoreSound();
        resetLiveBall();

        // Training happens only after a point.
        trainAfterPoint();

        updateGameUI();
    }
}

// ============================================================
// 9. DRAWING
// ============================================================

function drawArena() {
    const canvas = $("arena");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (
        canvas.width !== CONFIG.width ||
        canvas.height !== CONFIG.height
    ) {
        canvas.width = CONFIG.width;
        canvas.height = CONFIG.height;
    }

    ctx.clearRect(0, 0, CONFIG.width, CONFIG.height);

    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, CONFIG.width, CONFIG.height);

    // Center line
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 12]);

    ctx.beginPath();
    ctx.moveTo(0, CONFIG.height / 2);
    ctx.lineTo(CONFIG.width, CONFIG.height / 2);
    ctx.stroke();

    ctx.setLineDash([]);

    // AI paddle
    ctx.fillStyle = "#c7ff4d";
    ctx.fillRect(
        ai.x - CONFIG.paddleWidth / 2,
        ai.y - CONFIG.paddleHeight / 2,
        CONFIG.paddleWidth,
        CONFIG.paddleHeight
    );

    // Player paddle
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(
        player.x - CONFIG.paddleWidth / 2,
        player.y - CONFIG.paddleHeight / 2,
        CONFIG.paddleWidth,
        CONFIG.paddleHeight
    );

    // Ball
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

    // Scores
    ctx.fillStyle = "#c7ff4d";
    ctx.font = "bold 32px monospace";
    ctx.textAlign = "center";

    ctx.fillText(
        match.ai,
        CONFIG.width / 2,
        100
    );

    ctx.fillStyle = "#ffffff";

    ctx.fillText(
        match.player,
        CONFIG.width / 2,
        CONFIG.height - 85
    );

    // Labels
    ctx.font = "13px monospace";
    ctx.fillStyle = "#888";

    ctx.fillText("AI", 55, 40);
    ctx.fillText("PLAYER", 65, CONFIG.height - 25);
}

// ============================================================
// 10. GAME LOOP
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
// 11. GAME CONTROL
// ============================================================

function startMatch() {
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

    if (raf) cancelAnimationFrame(raf);

    lastFrameTime = performance.now();
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

    statistics.history.push({
        player: match.player,
        ai: match.ai,
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

function showScreen(name) {
    const screens = {
        menu: "menuScreen",
        game: "gameScreen",
        stats: "statsScreen"
    };

    Object.values(screens).forEach(id => {
        const element = $(id);
        if (element) element.classList.add("hidden");
    });

    const target = $(screens[name]);

    if (target) {
        target.classList.remove("hidden");
    }
}

function showPauseOverlay() {
    const overlay = $("pauseOverlay");
    if (overlay) overlay.classList.remove("hidden");
}

function hidePauseOverlay() {
    const overlay = $("pauseOverlay");
    if (overlay) overlay.classList.add("hidden");
}

// ============================================================
// 12. STATISTICS
// ============================================================

function getDefenseRate() {
    const total = statistics.playerPoints + statistics.aiPoints;

    if (total === 0) return 0;

    return (
        statistics.playerPoints / total
    ) * 100;
}

function updateGameUI() {
    const defenseRate = getDefenseRate();

    const liveRate = $("liveRate");
    const liveUpdates = $("liveUpdates");
    const liveSpeed = $("liveSpeed");

    if (liveRate) {
        liveRate.textContent =
            `${defenseRate.toFixed(1)}%`;
    }

    if (liveUpdates) {
        liveUpdates.textContent =
            agent ? agent.updates : 0;
    }

    if (liveSpeed) {
        liveSpeed.textContent =
            $("speedSelect")?.value || "NORMAL";
    }

    const statMatches = $("statMatches");
    const statRate = $("statRate");
    const statTime = $("statTime");
    const statUpdates = $("statUpdates");
    const statExperiences = $("statExperiences");
    const statEpsilon = $("statEpsilon");

    if (statMatches) {
        statMatches.textContent = statistics.matches;
    }

    if (statRate) {
        statRate.textContent =
            `${defenseRate.toFixed(1)}%`;
    }

    if (statTime) {
        statTime.textContent =
            formatTime(statistics.totalDuration);
    }

    if (statUpdates) {
        statUpdates.textContent =
            agent ? agent.updates : 0;
    }

    if (statExperiences) {
        statExperiences.textContent =
            agent ? agent.experiences : 0;
    }

    if (statEpsilon) {
        statEpsilon.textContent =
            agent ? agent.epsilon.toFixed(3) : "1.000";
    }

    drawStatsChart();
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
        const saved = localStorage.getItem(
            "pong-statistics"
        );

        if (saved) {
            const parsed = JSON.parse(saved);

            statistics = {
                ...statistics,
                ...parsed
            };
        }
    } catch (error) {
        console.error("Statistics load failed:", error);
    }
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

    if (history.length === 0) {
        ctx.fillStyle = "#777";
        ctx.font = "14px monospace";
        ctx.textAlign = "center";

        ctx.fillText(
            "NO MATCH DATA",
            width / 2,
            height / 2
        );

        return;
    }

    const maxScore = Math.max(
        1,
        ...history.map(item =>
            Math.max(item.player, item.ai)
        )
    );

    const step = width / Math.max(history.length, 1);

    ctx.strokeStyle = "#c7ff4d";
    ctx.lineWidth = 2;
    ctx.beginPath();

    history.forEach((item, index) => {
        const x = index * step + step / 2;
        const y = height - 20 -
            (item.player / maxScore) * (height - 40);

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();

    ctx.strokeStyle = "#777";
    ctx.beginPath();

    history.forEach((item, index) => {
        const x = index * step + step / 2;
        const y = height - 20 -
            (item.ai / maxScore) * (height - 40);

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();
}

// ============================================================
// 13. MODEL STATUS
// ============================================================

function updateModelStatus(message) {
    const element = $("modelStatus");

    if (element) {
        element.textContent = message;
    }
}

// ============================================================
// 14. INPUT
// ============================================================

function setupInput() {
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

        if (event.code === "Space") {
            event.preventDefault();

            if (screen === "game") {
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

    document.addEventListener("blur", () => {
        keys.left = false;
        keys.right = false;
    });
}

// ============================================================
// 15. BUTTONS
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

    $("resetStatsBtn")?.addEventListener("click", () => {
        statistics = {
            matches: 0,
            playerPoints: 0,
            aiPoints: 0,
            totalDuration: 0,
            updates: 0,
            experiences: 0,
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
// 16. INITIALIZATION
// ============================================================

async function init() {
    try {
        updateModelStatus("INITIALIZING AI...");

        // CPU backend avoids WebGL compatibility problems.
        await tf.setBackend("cpu");
        await tf.ready();

        console.log(
            "TensorFlow backend:",
            tf.getBackend()
        );

        loadStats();

        agent = new DQNAgent();

        await agent.initialize();

        envs = new TrainingEnvironments(
            CONFIG.parallelEnvironments
        );

        setupInput();
        setupButtons();

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
