import MLN from 'ml-naivebayes'
import MLR from 'ml-random-forest';
import MLT from 'ml-cart'

import fs from 'fs'
import CSV from 'papaparse'

const sortedHeadersRelevance = [
  "polyuria",
  "polydipsia",
  "age",
  "gender",
  "sudden_weight_loss",
  "partial_paresis",
  "polyphagia",
  "irritability",
  "alopecia",
  "visual_blurring",
  "weakness",
  "muscle_stiffness",
  "genital_thrush",
  "obesity",
  "delayed_healing",
  "itching",
]

function train_test_split(X, y, testSize = 0.25, randomState = 42) {
  if (X.length !== y.length) {
    throw new Error("Features (X) and target (y) must have the same length.")
  }

  function seededRandom(seed) {
    const x = Math.sin(seed) * 10000
    return x - Math.floor(x)
  }
  const random = () => seededRandom(randomState++) - 0.5

  const indices = Array.from({ length: X.length }, (_, i) => i)
  indices.sort(random)

  const splitIndex = Math.floor(X.length * (1 - testSize))
  const trainIndices = indices.slice(0, splitIndex)
  const testIndices = indices.slice(splitIndex)

  const trainX = trainIndices.map((i) => X[i])
  const testX = testIndices.map((i) => X[i])
  const trainY = trainIndices.map((i) => y[i])
  const testY = testIndices.map((i) => y[i])

  return { trainX, testX, trainY, testY }
}

function calculateRecall(yTrue, yPred) {
  let truePositives = 0;
  let falseNegatives = 0;

  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1) {
      if (yPred[i] === 1) {
        truePositives++;
      } else {
        falseNegatives++;
      }
    }
  }

  return truePositives / (truePositives + falseNegatives);
}


async function main(name, model) {
  const csvData = fs.readFileSync("./dataset-full.csv", "utf-8")
  const { data: rawData } = CSV.parse(csvData, { header: true })
  const data = rawData.map((row) => {
    const cleanedRow = {}
    for (const [key, value] of Object.entries(row)) {
      const newKey = key.trim().toLowerCase().replace(/ /g, "_").replace(/[()]/g, "")
      cleanedRow[newKey] = value
    }
    return cleanedRow
  }).filter((row) => Object.keys(row).length > 0)

  const positiveValues = ["yes", "positive", "Yes", "Positive", "Female", "female"]
  const allColumns = [...sortedHeadersRelevance, "class"]

  allColumns.forEach((column) => {
    data.forEach((row) => {
      row[column] = positiveValues.includes(row[column]) ? 1 : 0
    })
  })

  const X = data.map((row) => sortedHeadersRelevance.map((col) => row[col]))
  const y = data.map((row) => row["class"])

  const { trainX, testX, trainY, testY } = train_test_split(X, y, 0.3, 42)

  model.train(trainX, trainY)

  const yPred = model.predict(testX)

  console.log("Predictions:", yPred)

  const recall = calculateRecall(testY, yPred);
  console.log("Recall:", recall);

  const exportModel = model.toJSON()
  fs.writeFileSync(`../site/src/assets/model-${name}.json`, JSON.stringify(exportModel, null, 2))
}


Promise.all([
  main('naive', new MLN.GaussianNB()),
  main('random', new MLR.RandomForestClassifier()),
  main('tree', new MLT.DecisionTreeClassifier({
    gainFunction: 'gini',
    minNumSamples: 2
  }))
])
.catch(console.error)
.then(() => console.log('Done!'))
