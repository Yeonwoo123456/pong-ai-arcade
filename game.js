
/* ==========================================
   PONG AI ARCADE
   Browser-based DQN Reinforcement Learning
   ========================================== */

const $ = (id) => document.getElementById(id);

const W = 900;
const H = 620;

const PADDLE_W = 132;
const PADDLE_H = 12;
const BALL_R = 8;

const TOP = 34;
const BOTTOM = H - 34;

const STORAGE_KEY = "pong-ai-arcade-v1";
const MODEL_KEY = "pong-ai-dqn-v1";

let screen = "menu";
let paused = false;
let raf = 0;
let lastTime = 0;

let speed = 10;
let agent;
let envs;
let live;
let stats;
let match;
let keys = {};

let saving = false;

const defaultStats = () => ({
    matches: 0,
    hits: 0,
    attempts: 0,
    seconds: 0,
    experiences: 0,
    history: []
});

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function reflect(x) {
    const span = W - 2 * BALL_R;
    let q = (x - BALL_R) % (2 * span);

    if (q < 0) q += 2 * span;

    return BALL_R + (
        q <= span ? q : 2 * span - q
    );
}

function loadStats() {
    try {
        return {
            ...defaultStats(),
            ...JSON.parse(
                localStorage.getItem(STORAGE_KEY) || "{}"
            )
        };
    } catch {
        return defaultStats();
    }
}

function saveStats() {
    try {
        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify(stats)
        );
    } catch (error) {
        console.warn("Statistics save failed:", error);
    }
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);

    return (
        String(m).padStart(2, "0") +
        ":" +
        String(s).padStart(2, "0")
    );
}

function show(name) {
    screen = name;

    [
        "menuScreen",
        "gameScreen",
        "statsScreen"
    ].forEach(id => {
        $(id).classList.add("hidden");
    });

    $(`${name}Screen`).classList.remove("hidden");

    if (name === "stats") {
        drawStats();
    }
}


/* ==========================================
   DQN AGENT
   ========================================== */

class DQN {

    constructor() {
        this.model = this.makeModel();
        this.target = this.makeModel();

        this.copyWeights();

        this.optimizer = tf.train.adam(0.0005);

        this.memory = [];
        this.maxMemory = 30000;

        this.gamma = 0.985;
        this.batch = 64;

        this.epsilon = 1.0;
        this.minEpsilon = 0.04;
        this.decay = 0.9996;

        this.updates = 0;
        this.experiences = 0;
    }

    makeModel() {
        const model = tf.sequential();

        model.add(
            tf.layers.dense({
                inputShape: [7],
                units: 96,
                activation: "relu"
            })
        );

        model.add(
            tf.layers.dense({
                units: 96,
                activation: "relu"
            })
        );

        model.add(
            tf.layers.dense({
                units: 3
            })
        );

        return model;
    }

    copyWeights() {
        const weights = this.model.getWeights();

        this.target.setWeights(weights);

        weights.forEach(w => w.dispose());
    }

    act(state, explore = true) {
        if (
            explore &&
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
        if (this.memory.length >= this.maxMemory) {
            this.memory.shift();
        }

        this.memory.push([
            state,
            action,
            reward,
            nextState,
            done
        ]);

        this.experiences++;
    }

    async learn(repeats = 1) {
        if (this.memory.length < this.batch) {
            return;
        }

        for (let n = 0; n < repeats; n++) {

            const batch = [];

            for (let i = 0; i < this.batch; i++) {
                const index = Math.floor(
                    Math.random() * this.memory.length
                );

                batch.push(this.memory[index]);
            }

            const states = batch.map(x => x[0]);
            const nextStates = batch.map(x => x[3]);

            const actions = batch.map(x => x[1]);
            const rewards = batch.map(x => x[2]);
            const dones = batch.map(x => x[4] ? 1 : 0);

            const xs = tf.tensor2d(states);
            const nxs = tf.tensor2d(nextStates);

            const rewardTensor = tf.tensor1d(rewards);
            const doneTensor = tf.tensor1d(dones);

            const nextQ = this.target.predict(nxs);

            const maxNextQ = nextQ.max(1);

            const targets = tf.tidy(() => {
                return rewardTensor.add(
                    maxNextQ.mul(this.gamma).mul(
                        tf.scalar(1).sub(doneTensor)
                    )
                );
            });

            const currentQ = this.model.predict(xs);

            const currentArray = currentQ.arraySync();

            const targetArray = targets.dataSync();

            for (let i = 0; i < this.batch; i++) {
                currentArray[i][actions[i]] = targetArray[i];
            }

            const ys = tf.tensor2d(currentArray);

            const loss = this.optimizer.minimize(() => {
                const predictions = this.model.predict(xs);

                return tf.losses.huberLoss(
                    ys,
                    predictions
                ).mean();

            }, true);

            if (loss) loss.dispose();

            tf.dispose([
                xs,
                nxs,
                rewardTensor,
                doneTensor,
                nextQ,
                maxNextQ,
                targets,
                currentQ,
                ys
            ]);

            this.epsilon = Math.max(
                this.minEpsilon,
                this.epsilon * this.decay
            );

            this.updates++;

            if (this.updates % 180 === 0) {
                this.copyWeights();
            }
        }
    }

    async save() {
        if (saving) return;

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

        } catch (error) {
            console.warn("Model save failed:", error);
        }

        saving = false;
    }

    async load() {
        try {
            const restored = await tf.loadLayersModel(
                `localstorage://${MODEL_KEY}`
            );

            this.model.dispose();
            this.target.dispose();

            this.model = restored;
            this.target = this.makeModel();

            this.copyWeights();

            this.epsilon = Number(
                localStorage.getItem("pong-ai-epsilon") || 1
            );

            this.updates = Number(
                localStorage.getItem("pong-ai-updates") || 0
            );

            this.experiences = Number(
                localStorage.getItem("pong-ai-experiences") || 0
            );

            return true;

        } catch {
            return false;
        }
    }
}


/* ==========================================
   PARALLEL TRAINING ENVIRONMENTS
   ========================================== */

class ParallelWorlds {

    constructor(count = 32) {
        this.n = count;
        this.resetAll();
    }

    resetOne(i) {
        this.x[i] = 100 + Math.random() * (W - 200);

        this.y[i] = H * 0.4 + Math.random() * H * 0.35;

        this.vx[i] =
            (Math.random() < 0.5 ? -1 : 1) *
            (2 + Math.random() * 3);

        this.vy[i] = -(3 + Math.random() * 3);

        this.p[i] = 100 + Math.random() * (W - 200);

        this.age[i] = 0;
    }

    resetAll() {
        this.x = [];
        this.y = [];
        this.vx = [];
        this.vy = [];
        this.p = [];
        this.age = [];

        for (let i = 0; i < this.n; i++) {
            this.resetOne(i);
        }
    }

    state(i) {
        const distance = Math.max(
            0,
            this.y[i] - (TOP + PADDLE_H + BALL_R)
        );

        const t = distance / Math.max(0.1, -this.vy[i]);

        const intercept = reflect(
            this.x[i] + this.vx[i] * t
        );

        return [
            this.x[i] / W,
            this.y[i] / H,
            this.vx[i] / 7,
            this.vy[i] / 7,
            this.p[i] / W,
            intercept / W,
            (intercept - this.p[i]) / W
        ];
    }

    step(actions) {
        const transitions = [];

        for (let i = 0; i < this.n; i++) {

            const oldState = this.state(i);

            const oldDistance = Math.abs(oldState[6]);

            // 0 = LEFT, 1 = STAY, 2 = RIGHT
            this.p[i] = clamp(
                this.p[i] + (actions[i] - 1) * 6,
                PADDLE_W / 2,
                W - PADDLE_W / 2
            );

            this.x[i] += this.vx[i];
            this.y[i] += this.vy[i];

            // Side wall collision
            if (
                this.x[i] < BALL_R ||
                this.x[i] > W - BALL_R
            ) {
                this.vx[i] *= -1;

                this.x[i] = clamp(
                    this.x[i],
                    BALL_R,
                    W - BALL_R
                );
            }

            const topLine = this.y[i] <=
                TOP + PADDLE_H + BALL_R;

            const hit = topLine &&
                Math.abs(this.x[i] - this.p[i]) <=
                PADDLE_W / 2 + BALL_R;

            const missed = this.y[i] <= TOP + BALL_R;

            let reward =
                -0.002 -
                0.012 * Math.min(oldDistance, 1);

            if (hit) {
                reward += 1;

                this.y[i] = TOP + PADDLE_H + BALL_R + 1;

                this.vy[i] = Math.abs(this.vy[i]) * 1.01;

                this.vx[i] = clamp(
                    this.vx[i] +
                    (this.x[i] - this.p[i]) * 0.025,
                    -7,
                    7
                );
            }

            if (missed && !hit) {
                reward -= 1;
            }

            this.age[i]++;

            if (this.y[i] > H - BALL_R) {
                this.y[i] = H - BALL_R;
                this.vy[i] *= -1;
            }

            const done = missed || this.age[i] > 850;

            const nextState = this.state(i);

            transitions.push([
                oldState,
                actions[i],
                reward,
                nextState,
                done
            ]);

            if (done) {
                this.resetOne(i);
            }
        }

        return transitions;
    }
}


/* ==========================================
   LIVE GAME
   ========================================== */

function resetLiveBall() {
    live.x = W / 2;
    live.y = H / 2;

    live.vx =
        (Math.random() < 0.5 ? -1 : 1) * 4;

    live.vy =
        (Math.random() < 0.5 ? -1 : 1) * 4.2;
}

function newMatch() {
    match = {
        player: 0,
        ai: 0,
        hits: 0,
        attempts: 0,
        seconds: 0,
        started: performance.now(),
        pausedDuration: 0,
        pauseStarted: null,
        historySaved: false
    };

    live = {
        px: W / 2,
        ax: W / 2,
        x: W / 2,
        y: H / 2,
        vx: 4,
        vy: 4.2
    };

    resetLiveBall();

    paused = false;

    $("pauseOverlay").classList.add("hidden");
    $("pauseBtn").textContent = "Ⅱ PAUSE";

    show("game");
}

function getMatchSeconds() {
    if (!match) return 0;

    const now = paused && match.pauseStarted
        ? match.pauseStarted
        : performance.now();

    return Math.max(
        0,
        (now - match.started - match.pausedDuration) / 1000
    );
}

function finishMatch() {
    if (
        screen !== "game" ||
        !match ||
        match.historySaved
    ) {
        return;
    }

    match.seconds = getMatchSeconds();

    stats.matches++;

    stats.hits += match.hits;
    stats.attempts += match.attempts;
    stats.seconds += match.seconds;

    const rate = match.attempts
        ? 100 * match.hits / match.attempts
        : 0;

    stats.history.push({
        rate,
        duration: match.seconds,
        updates: agent.updates
    });

    stats.history = stats.history.slice(-30);

    stats.experiences = agent.experiences;

    match.historySaved = true;

    saveStats();
    agent.save();

    cancelAnimationFrame(raf);

    show("menu");
}

function liveState() {
    const distance = Math.max(
        0,
        live.y - (TOP + PADDLE_H + BALL_R)
    );

    const direction = live.vy < 0 ? 1 : -1;

    const t = distance / Math.max(
        0.1,
        Math.abs(live.vy)
    );

    const intercept = reflect(
        live.x + live.vx * t * direction
    );

    return [
        live.x / W,
        live.y / H,
        live.vx / 7,
        live.vy / 7,
        live.ax / W,
        intercept / W,
        (intercept - live.ax) / W
    ];
}

function moveLive() {

    // Human player: bottom paddle
    if (keys.ArrowLeft || keys.a || keys.A) {
        live.px -= 8;
    }

    if (keys.ArrowRight || keys.d || keys.D) {
        live.px += 8;
    }

    live.px = clamp(
        live.px,
        PADDLE_W / 2,
        W - PADDLE_W / 2
    );

    // AI: top paddle
    const action = agent.act(
        liveState(),
        false
    );

    live.ax = clamp(
        live.ax + (action - 1) * 7.5,
        PADDLE_W / 2,
        W - PADDLE_W / 2
    );

    // Move ball
    live.x += live.vx;
    live.y += live.vy;

    // Left and right walls
    if (
        live.x < BALL_R ||
        live.x > W - BALL_R
    ) {
        live.vx *= -1;

        live.x = clamp(
            live.x,
            BALL_R,
            W - BALL_R
        );
    }

    // AI paddle collision at the TOP
    if (
        live.vy < 0 &&
        live.y - BALL_R <= TOP + PADDLE_H
    ) {
        match.attempts++;

        if (
            Math.abs(live.x - live.ax) <=
            PADDLE_W / 2 + BALL_R
        ) {
            match.hits++;

            live.y = TOP + PADDLE_H + BALL_R;

            live.vy = Math.abs(live.vy) * 1.02;

            live.vx = clamp(
                live.vx +
                (live.x - live.ax) * 0.035,
                -7,
                7
            );

        } else {
            match.player++;

            resetLiveBall();
        }
    }

    // Human paddle collision at the BOTTOM
    if (
        live.vy > 0 &&
        live.y + BALL_R >= BOTTOM
    ) {
        if (
            Math.abs(live.x - live.px) <=
            PADDLE_W / 2 + BALL_R
        ) {
            live.y = BOTTOM - BALL_R;

            live.vy = -Math.abs(live.vy) * 1.02;

            live.vx = clamp(
                live.vx +
                (live.x - live.px) * 0.035,
                -7,
                7
            );

        } else {
            match.ai++;

            resetLiveBall();
        }
    }

    match.seconds = getMatchSeconds();
}


/* ==========================================
   CANVAS RENDERING
   ========================================== */

function drawArena() {
    const canvas = $("arena");
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#050806";
    ctx.fillRect(0, 0, W, H);

    // Outer border
    ctx.strokeStyle = "#dce8dc";
    ctx.lineWidth = 2;

    ctx.strokeRect(
        8,
        8,
        W - 16,
        H - 16
    );

    // Center line
    ctx.setLineDash([10, 13]);

    ctx.strokeStyle = "#536157";

    ctx.beginPath();
    ctx.moveTo(W / 2, 12);
    ctx.lineTo(W / 2, H - 12);
    ctx.stroke();

    ctx.setLineDash([]);

    // AI paddle at TOP
    ctx.fillStyle = "#edf4e9";

    ctx.fillRect(
        live.ax - PADDLE_W / 2,
        TOP,
        PADDLE_W,
        PADDLE_H
    );

    // Player paddle at BOTTOM
    ctx.fillRect(
        live.px - PADDLE_W / 2,
        BOTTOM - PADDLE_H,
        PADDLE_W,
        PADDLE_H
    );

    // Ball
    ctx.fillStyle = "#c7ff4d";

    ctx.fillRect(
        live.x - BALL_R,
        live.y - BALL_R,
        BALL_R * 2,
        BALL_R * 2
    );

    // Score
    ctx.fillStyle = "#c7ff4d";
    ctx.font = "bold 17px monospace";

    ctx.fillText(
        `${match.player} : ${match.ai}`,
        W / 2 - 28,
        H / 2 - 15
    );

    // Labels
    ctx.fillStyle = "#75e6b2";
    ctx.font = "12px monospace";

    ctx.fillText(
        "AI — TOP PADDLE",
        22,
        TOP + 10
    );

    ctx.fillStyle = "#ffcf75";

    ctx.fillText(
        "PLAYER — BOTTOM PADDLE",
        22,
        BOTTOM - 18
    );
}


/* ==========================================
   GAME UI
   ========================================== */

function updateGameUI() {
    $("liveRate").textContent =
        match.attempts
            ? `${(
                100 * match.hits / match.attempts
            ).toFixed(1)}%`
            : "—";

    $("liveUpdates").textContent =
        agent.updates.toLocaleString();

    $("liveSpeed").textContent = `${speed}×`;
}


/* ==========================================
   TRAINING LOOP
   ========================================== */

async function gameLoop(now) {
    if (screen !== "game") return;

    const dt = Math.min(
        50,
        now - lastTime || 16
    );

    lastTime = now;

    if (!paused) {

        /*
         * Parallel simulated training.
         * Each speed unit performs one training
         * step across all simulated environments.
         */

        for (let k = 0; k < speed; k++) {

            const states = envs.x.map(
                (_, i) => envs.state(i)
            );

            const actions = states.map(
                state => agent.act(state, true)
            );

            const transitions = envs.step(actions);

            transitions.forEach(transition => {
                agent.remember(...transition);
            });

            if (agent.memory.length >= agent.batch) {
                await agent.learn(1);
            }
        }

        moveLive();

        updateGameUI();

        // Save model periodically
        if (
            agent.updates > 0 &&
            agent.updates % 100 === 0
        ) {
            agent.save();
        }
    }

    drawArena();

    raf = requestAnimationFrame(gameLoop);
}


/* ==========================================
   STATISTICS
   ========================================== */

function drawStats() {
    const rate = stats.attempts
        ? 100 * stats.hits / stats.attempts
        : 0;

    $("statMatches").textContent =
        stats.matches.toLocaleString();

    $("statRate").textContent =
        `${rate.toFixed(1)}%`;

    $("statTime").textContent =
        formatTime(stats.seconds);

    $("statUpdates").textContent =
        agent.updates.toLocaleString();

    $("statExperiences").textContent =
        agent.experiences.toLocaleString();

    $("statEpsilon").textContent =
        agent.epsilon.toFixed(3);

    drawChart();
}

function drawChart() {
    const canvas = $("statsChart");
    const ctx = canvas.getContext("2d");

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    ctx.font = "13px monospace";

    const left = 48;
    const right = w - 20;
    const top = 28;
    const bottom = h - 35;

    ctx.strokeStyle = "#34413b";
    ctx.fillStyle = "#829087";
    ctx.lineWidth = 1;

    [0, 25, 50, 75, 100].forEach(value => {
        const y = bottom -
            (value / 100) * (bottom - top);

        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();

        ctx.fillText(
            value,
            left - 35,
            y + 4
        );
    });

    const values = stats.history.slice(-20);

    if (!values.length) {
        ctx.fillText(
            "NO MATCH DATA — PLAY A MATCH FIRST",
            w / 2 - 160,
            h / 2
        );

        return;
    }

    const points = values.map((value, i) => {
        return {
            x: values.length === 1
                ? (left + right) / 2
                : left + i * (right - left) /
                    (values.length - 1),

            y: bottom -
                clamp(value.rate, 0, 100) / 100 *
                (bottom - top)
        };
    });

    ctx.strokeStyle = "#c7ff4d";
    ctx.lineWidth = 3;

    ctx.beginPath();

    points.forEach((point, i) => {
        if (i === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });

    ctx.stroke();

    points.forEach(point => {
        ctx.fillStyle = "#edf4e9";

        ctx.fillRect(
            point.x - 4,
            point.y - 4,
            8,
            8
        );
    });
}


/* ==========================================
   PAUSE
   ========================================== */

function togglePause() {
    if (screen !== "game") return;

    if (!paused) {
        paused = true;
        match.pauseStarted = performance.now();

    } else {
        paused = false;

        if (match.pauseStarted) {
            match.pausedDuration +=
                performance.now() - match.pauseStarted;
        }

        match.pauseStarted = null;
    }

    $("pauseOverlay").classList.toggle(
        "hidden",
        !paused
    );

    $("pauseBtn").textContent = paused
        ? "▶ RESUME"
        : "Ⅱ PAUSE";
}


/* ==========================================
   INITIALIZATION
   ========================================== */

async function init() {
    stats = loadStats();

    $("modelStatus").textContent =
        "AI MODEL: LOADING";

    try {
        await tf.ready();

        agent = new DQN();

        const restored = await agent.load();

        envs = new ParallelWorlds(32);

        $("modelStatus").textContent = restored
            ? "AI MODEL: RESTORED"
            : "AI MODEL: NEW";

        $("modelStatus").nextElementSibling.style.background =
            "#c7ff4d";

    } catch (error) {
        console.error(error);

        $("modelStatus").textContent =
            "AI MODEL: ERROR";

        return;
    }

    // PLAY
    $("playBtn").addEventListener("click", () => {
        newMatch();

        lastTime = performance.now();

        cancelAnimationFrame(raf);

        raf = requestAnimationFrame(gameLoop);
    });

    // STATISTICS
    $("statsBtn").addEventListener("click", () => {
        show("stats");
        drawStats();
    });

    $("statsBackBtn").addEventListener(
        "click",
        () => show("menu")
    );

    // MENU
    $("gameMenuBtn").addEventListener(
        "click",
        finishMatch
    );

    $("endMatchBtn").addEventListener(
        "click",
        finishMatch
    );

    // NEW MATCH
    $("newMatchBtn").addEventListener("click", () => {
        finishMatch();

        newMatch();

        lastTime = performance.now();

        cancelAnimationFrame(raf);

        raf = requestAnimationFrame(gameLoop);
    });

    // PAUSE
    $("pauseBtn").addEventListener(
        "click",
        togglePause
    );

    // TRAINING SPEED
    $("speedSelect").addEventListener("change", event => {
        speed = Number(event.target.value);
    });

    // RESET DATA
    $("resetStatsBtn").addEventListener(
        "click",
        async () => {
            const confirmed = confirm(
                "이 브라우저에 저장된 통계와 AI 모델을 모두 초기화할까요?"
            );

            if (!confirmed) return;

            localStorage.removeItem(STORAGE_KEY);

            await tf.io.removeModel(
                `localstorage://${MODEL_KEY}`
            ).catch(() => {});

            localStorage.removeItem("pong-ai-epsilon");
            localStorage.removeItem("pong-ai-updates");
            localStorage.removeItem("pong-ai-experiences");

            location.reload();
        }
    );

    // KEYBOARD
    window.addEventListener("keydown", event => {
        keys[event.key] = true;

        if (
            event.code === "Space" &&
            screen === "game"
        ) {
            event.preventDefault();
            togglePause();
        }
    });

    window.addEventListener("keyup", event => {
        keys[event.key] = false;
    });

    // SAVE WHEN LEAVING
    window.addEventListener("beforeunload", () => {
        if (stats) saveStats();

        if (agent) {
            agent.save();
        }
    });

    show("menu");
}

init();
