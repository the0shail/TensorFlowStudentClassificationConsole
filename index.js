// Задача:
// Обучите нейронную сеть для классификации студентов на два класса: сдал или не сдал экзамен.

// | Подготовка (часы)   | Сон перед экзаменом (часы)  | Сдал (1) / Не сдал (0) |
// |---------------------|-----------------------------|------------------------|
// |         5           |             8               |           0            |
// |         7           |             7               |           1            |
// |         2           |             9               |           0            |
// |         10          |             6               |           1            |

import tf from "@tensorflow/tfjs";

const INPUT_TEST = tf.tensor2d(
    [
        [0, 0],
        [8, 6]
    ]
);
const LABEL_TEST = tf.tensor1d(
    [0, 1]
);

const model = tf.sequential({
    layers: [
        tf.layers.dense({units: 64, inputShape: [2], activation: 'relu'}),
        // Добавляем слой регуляризации для предотвращения переобучения
        tf.layers.dropout({rate: 0.3}),
        tf.layers.dense({units: 32, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'})
    ]
});

const {inputs, labels} = randomTest(1000);

const xs = tf.tensor2d(inputs);
const ys = tf.tensor1d(labels);

model.compile({loss: "binaryCrossentropy", optimizer: 'adam', metrics: ['accuracy']});

model.fit(xs, ys, {
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}`, logs);
        }
    }
}).then(() => {
    let result = model.predict(INPUT_TEST);
    result.print();

    const evaluation = model.evaluate(INPUT_TEST, LABEL_TEST);
    console.log('Точность на тестовых данных:', evaluation[1].dataSync()[0]);
})


function randomTest(count = 10) {
    let inputs = [];
    let labels = [];

    for (let i = 0; i < count; i++) {
        let learn = Math.random() * 10;
        let sleep = Math.random() * 10;

        let success = (learn + sleep) > 10 ? 1 : 0;

        inputs[i] = [learn, sleep];
        labels[i] = success;
    }

    return {inputs, labels};
}
