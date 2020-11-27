const width = 8;
const height = 8;

var game = document.getElementById("game");
const cells = [];

for (let i = 0; i < width * height; i++) {
  const cell = document.createElement("div");
  cell.classList.add("cell");
  game.appendChild(cell);
  cells.push(cell);
}

function random(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

function buildState(food, snake) {
  const buffer = new Array(width * height).fill(0);
  buffer[food.y * width + food.x] = -1;
  for (let i = 0; i < snake.length; i++) {
    const cord = snake[i];
    buffer[cord.y * width + cord.x] = 1;
  }
  return tf.tensor([buffer], [1, width, height]);
}

function cellAt({ x, y }) {
  return cells[y * width + x];
}

function draw(state) {
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const value = state.dataSync()[y * width + x];
      cellAt({ x, y }).classList = ["cell"];
      if (value === -1) {
        cellAt({ x, y }).classList.add("food");
      } else if (value === 1) {
        cellAt({ x, y }).classList.add("snake");
      }
    }
  }
}

const timer = (ms) => new Promise((res) => setTimeout(res, ms));

function copy(x) {
  return Object.assign({}, x);
}

function placeFood(snake) {
  const cord = { x: random(0, width), y: random(0, height) };
  for (let i = 0; i < snake.length; i++) {
    if (snake[i].x === cord.x && snake[i].y === cord.y) {
      return placeFood(snake);
    }
  }
  return cord;
}

async function step(action, food, snake) {
  const previousState = buildState(food, [...snake.map((s) => copy(s))]);
  const { x, y } = [...snake.map((s) => copy(s))][snake.length - 1];
  var done = false;

  // Compute new position
  var newPos = { x: -1, y: -1 };
  switch (action) {
    case 0:
      if (y - 1 < 0) {
        done = true;
      } else {
        newPos = { x: x, y: y - 1 };
      }
      break;
    case 1:
      if (x + 1 >= width) {
        done = true;
      } else {
        newPos = { x: x + 1, y: y };
      }
      break;
    case 2:
      if (y + 1 >= height) {
        done = true;
      } else {
        newPos = { x: x, y: y + 1 };
      }
      break;
    case 3:
      if (x - 1 < 0) {
        done = true;
      } else {
        newPos = { x: x - 1, y: y };
      }
      break;
    default:
      console.log("Got unknown action", action);
      return;
  }

  // Check if new pos is in snake
  for (let i = 0; i < snake.length; i++) {
    const cord = snake[i];
    if (newPos.x === cord.x && newPos.y === cord.y) {
      done = true;
    }
  }

  // Check if food was eaten
  var reward = done ? -1 : 0;
  if (!done) {
    if (newPos.x === food.x && newPos.y === food.y) {
      reward = 1;
      food = placeFood(snake);
    } else {
      snake.shift();
    }
    snake.push(newPos);
  }

  const newState = buildState(food, snake);
  draw(newState);

  return {
    state: tf.stack([previousState, newState], -1),
    done: done,
    reward: reward,
    snake: snake,
    food: food,
  };
}

(async () => {
  const model = await tf.loadLayersModel("./tfjs_model/model.json");

  const snakeX = random(0, width);
  const snakeY = random(3, height);
  var totalReward = 3;

  var snake = [
    { x: snakeX, y: snakeY },
    { x: snakeX, y: snakeY - 1 },
    { x: snakeX, y: snakeY - 2 },
  ];
  var food = placeFood(snake);

  var action = 0;
  while (true) {
    const {
      state,
      done,
      reward,
      snake: newSnake,
      food: newFood,
    } = await step(action, copy(food), [...snake.map((s) => copy(s))]);
    snake = newSnake;
    food = newFood;
    if (done) {
      //   window.location.reload();
      break;
    }
    totalReward += reward;
    document.getElementById("total-reward").textContent = totalReward;
    const preds = await model.predict(state);

    const visualizeQ = false;
    if (visualizeQ) {
      for (let index = 0; index < cells.length; index++) {
        cells[index].textContent = "";
      }

      const { x, y } = snake[snake.length - 1];
      cellAt({ x: x, y: y }).textContent = ":)";
      if (y - 1 >= 0) {
        cellAt({ x: x, y: y - 1 }).textContent = Number.parseFloat(
          preds.dataSync()[0]
        ).toPrecision(2);
      }
      if (x + 1 < width) {
        cellAt({ x: x + 1, y: y }).textContent = Number.parseFloat(
          preds.dataSync()[1]
        ).toPrecision(2);
      }
      if (y + 1 < height) {
        cellAt({ x: x, y: y + 1 }).textContent = Number.parseFloat(
          preds.dataSync()[2]
        ).toPrecision(2);
      }
      if (x - 1 >= 0) {
        cellAt({ x: x - 1, y: y }).textContent = Number.parseFloat(
          preds.dataSync()[3]
        ).toPrecision(2);
      }
    }

    action = tf.argMax(preds, -1).dataSync()[0];
    await timer(250);
  }
})();
