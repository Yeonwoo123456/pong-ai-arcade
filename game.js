
"use strict";

/* =====================================================
   PONG AI - NEURAL ARCADE
   Weak initial prediction + gradual AI progression
   ===================================================== */

const CONFIG = {
    width: 900,
    height: 620,

    ballRadius: 9,
    ballSpeed: 5,

    playerSpeed: 8,
    playerWidth: 125,
    paddleHeight: 13,

    aiBaseSpeed: 4.5,
    aiMaxSpeed: 9.5,

    aiBaseWidth: 88,
    aiMaxWidth: 135,

    // AI starts deliberately inaccurate.
    initialErrorRate: 1.00,
    finalErrorRate: 0.04,

    initialErrorRange: 360,
    finalErrorRange: 12,

    maxLevel: 30,

    epsilonStart: 0.35,
    epsilonMin: 0.03,
    epsilonDecay: 0.995
};

const STORAGE = {
    stats: "pong-statistics",
    attempts: "pong-ai-attempts-v12"
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


/* =====================================================
   STORAGE
   ===================================================== */

function storageGet(key, fallback = null) {
    try {
        const value = localStorage.getItem(key);
        return value === null ? fallback : value;
    } catch {
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

function loadNumber(key, fallback = 0) {
    const value = Number(storageGet(key, fallback));
    return Number.isFinite(value) ? value : fallback;
}


/* =====================================================
   GAME STATE
   ===================================================== */

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

let servePredictionOffset = 0;
let servePredictionFails = true;
let pointFinished = false;


/* =====================================================
   AI PROGRESSION
   ===================================================== */

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

function advanceAISkill() {
    aiAttempts++;

    updateAISkill();

    storageSet(
        STORAGE.attempts,
        String(aiAttempts)
    );

    updateUI();
}


/* =====================================================
   STATISTICS
   ===================================================== */

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


/* =====================================================
   SCREEN MANAGEMENT
   ===================================================== */

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


/* =====================================================
   SCOREBOARD
   ===================================================== */

function updateScoreboard() {
    if (ui.aiScoreDisplay) {
        ui.aiScoreDisplay.textContent = aiScore;
    }

    if (ui.playerScoreDisplay) {
        ui.playerScoreDisplay.textContent = playerScore;
    }
}


/* =====================================================
   UI UPDATE
   ===================================================== */

function updateUI() {
    updateAISkill();

    const accuracy = Math.round(
        (1 - getAIErrorRate()) * 100
    );

    if (ui.liveRate) {
        ui.liveRate.textContent = `${accuracy}%`;
    }

    if (ui.liveUpdates) {
        ui.liveUpdates.textContent = aiAttempts;
    }

    if (ui.liveSpeed) {
        ui.liveSpeed.textContent = `LV.${aiLevel}`;
    }

    if (ui.modelStatus) {
        ui.modelStatus.textContent = "AI ACTIVE";
    }

    if (ui.statMatches) {
        ui.statMatches.textContent =
            statistics.matches;
    }

    if (ui.statRate) {
        ui.statRate.textContent =
            `${getWinRate()}%`;
    }

    if (ui.statTime) {
        ui.statTime.textContent =
            formatTime(statistics.totalDuration);
    }

    if (ui.statUpdates) {
        ui.statUpdates.textContent = aiAttempts;
    }

    if (ui.statExperiences) {
        ui.statExperiences.textContent =
            totalExperiences;
    }

    if (ui.statEpsilon) {
        const epsilon = Math.max(
            CONFIG.epsilonMin,
            CONFIG.epsilonStart *
            Math.pow(CONFIG.epsilonDecay, aiAttempts)
        );

        ui.statEpsilon.textContent =
            epsilon.toFixed(3);
    }

    updateScoreboard();
}


/* =====================================================
   BALL PREDICTION
   ===================================================== */

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


/* =====================================================
   AI MOVEMENT
   ===================================================== */

function moveAI() {
    ai.width = getAIWidth();

    let targetX = CONFIG.width / 2;

    if (ball.vy < 0) {
        const prediction = predictBallXAtY(ai.y);

        if (servePredictionFails) {
            targetX =
                prediction + servePredictionOffset;
        } else {
            targetX = prediction;
        }
    }

    const minX = ai.width / 2;
    const maxX = CONFIG.width - ai.width / 2;

    targetX = Math.max(
        minX,
        Math.min(maxX, targetX)
    );

    const difference = targetX - ai.x;

    if (Math.abs(difference) > 1) {
        ai.x += Math.sign(difference) *
            Math.min(
                Math.abs(difference),
                getAISpeed()
            );
    }
}


/* =====================================================
   BALL RESET
   ===================================================== */

function resetBall() {
    ball.x = CONFIG.width / 2;
    ball.y = CONFIG.height / 2;

    ball.vx =
        (Math.random() < 0.5 ? -1 : 1) *
        (2 + Math.random() * 2);

    ball.vy =
        (Math.random() < 0.5 ? -1 : 1) *
        CONFIG.ballSpeed;

    // Select prediction error for each serve.
    servePredictionFails =
        Math.random() < getAIErrorRate();

    servePredictionOffset =
        (Math.random() * 2 - 1) *
        getAIErrorRange();
}


/* =====================================================
   COLLISION
   ===================================================== */

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

    totalExperiences++;

    if (isAI) {
        // AI successfully returned the ball.
        playSound(520);
    } else {
        playSound(650);
    }
}


/* =====================================================
   POINT HANDLING
   ===================================================== */

function finishPoint(playerWon) {
    if (pointFinished) return;

    pointFinished = true;

    if (playerWon) {
        playerScore++;
        statistics.playerPoints++;

        playSound(260, 0.15);
    } else {
        aiScore++;
        statistics.aiPoints++;

        playSound(800, 0.15);
    }

    totalExperiences++;

    // AI improves after every completed point.
    advanceAISkill();

    resetBall();

    updateUI();

    pointFinished = false;
}


/* =====================================================
   GAME PHYSICS
   ===================================================== */

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

    // Side walls.
    if (
        ball.x - ball.radius <= 0 ||
        ball.x + ball.radius >= CONFIG.width
    ) {
        ball.vx *= -1;
        playSound(350);
    }

    // AI paddle at the top.
    if (ball.vy < 0 && ballHitsPaddle(ai)) {
        ball.y =
            ai.y + ai.height / 2 + ball.radius;

        bounceFromPaddle(ai, true);
    }

    // Player paddle at the bottom.
    if (ball.vy > 0 && ballHitsPaddle(player)) {
        ball.y =
            player.y - player.height / 2 -
            ball.radius;

        bounceFromPaddle(player, false);
    }

    // Ball passes the player.
    if (ball.y - ball.radius > CONFIG.height) {
        finishPoint(true);
    }

    // Ball passes the AI.
    if (ball.y + ball.radius < 0) {
        finishPoint(false);
    }
}


/* =====================================================
   DRAWING
   ===================================================== */

function drawGame() {
    if (!ctx) return;

    ctx.clearRect(
        0,
        0,
        CONFIG.width,
        CONFIG.height
    );

    ctx.fillStyle = "#050505";
    ctx.fillRect(
        0,
        0,
        CONFIG.width,
        CONFIG.height
    );

    // Center line.
    ctx.strokeStyle = "#444";
    ctx.setLineDash([10, 12]);

    ctx.beginPath();
    ctx.moveTo(0, CONFIG.height / 2);
    ctx.lineTo(CONFIG.width, CONFIG.height / 2);
    ctx.stroke();

    ctx.setLineDash([]);

    // AI paddle.
    ctx.fillStyle = "#c7ff4d";

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

    ctx.arc(
        ball.x,
        ball.y,
        ball.radius,
        0,
        Math.PI * 2
    );

    ctx.fill();
}


/* =====================================================
   GAME LOOP
   ===================================================== */

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
                "GAME ERROR";
        }

        return;
    }

    animationFrame = requestAnimationFrame(gameLoop);
}


/* =====================================================
   MATCH CONTROL
   ===================================================== */

function startMatch() {
    playerScore = 0;
    aiScore = 0;

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

    updateUI();
    drawGame();

    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }

    gameLoop();
}

function pauseGame() {
    if (!running || paused) return;

    paused = true;

    if (ui.pauseOverlay) {
        ui.pauseOverlay.style.display = "flex";
    }
}

function resumeGame() {
    if (!running || !paused) return;

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
        animationFrame = null;
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

    if (ui.pauseOverlay) {
        ui.pauseOverlay.style.display = "none";
    }

    showScreen("menu");
    updateUI();
}


/* =====================================================
   CHART
   ===================================================== */

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

    chartCtx.strokeStyle = "#333";
    chartCtx.lineWidth = 1;

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
    drawLine("aiScore", "#c7ff4d");
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


/* =====================================================
   AUDIO
   ===================================================== */

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

        const oscillator =
            audioContext.createOscillator();

        const gain =
            audioContext.createGain();

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


/* =====================================================
   KEYBOARD CONTROLS
   ===================================================== */

document.addEventListener("keydown", event => {
    const key = event.key.toLowerCase();

    if (
        key === "arrowleft" ||
        key === "arrowright"
    ) {
        event.preventDefault();
    }

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

window.addEventListener("blur", () => {
    keys.left = false;
    keys.right = false;
});


/* =====================================================
   BUTTON EVENTS
   ===================================================== */

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
        ui.statsBack.addEventListener("click", () => {
            showScreen("menu");
        });
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

            totalExperiences = 0;
            totalUpdates = 0;

            saveStatistics();
            updateUI();
            drawChart();
        });
    }
}


/* =====================================================
   INITIALIZATION
   ===================================================== */

function init() {
    try {
        if (!canvas || !ctx) {
            console.error("Missing #arena canvas.");
            return;
        }

        canvas.width = CONFIG.width;
        canvas.height = CONFIG.height;

        loadStatistics();
        updateAISkill();

        setupButtons();
        setupChartHover();

        showScreen("menu");

        updateUI();
        drawChart();

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
