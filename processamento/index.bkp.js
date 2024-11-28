import fs from "fs"
import { GaussianNB, StandardScaler, createPythonBridge } from "sklearn"
import CSV from "papaparse"

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
    throw new Error("Features (X) and target (y) must have the same length.");
  }

  // Seed the random number generator for reproducibility
  function seededRandom(seed) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }
  const random = (a, b) => seededRandom(randomState++) - 0.5;

  // Shuffle the indices
  const indices = Array.from({ length: X.length }, (_, i) => i);
  indices.sort(random);

  // Calculate the split index
  const splitIndex = Math.floor(X.length * (1 - testSize));

  // Split into training and testing sets
  const trainIndices = indices.slice(0, splitIndex);
  const testIndices = indices.slice(splitIndex);

  const trainX = trainIndices.map((i) => X[i]);
  const testX = testIndices.map((i) => X[i]);
  const trainY = trainIndices.map((i) => y[i]);
  const testY = testIndices.map((i) => y[i]);

  return { trainX, testX, trainY, testY };
}


async function main() {

  const csvData = fs.readFileSync("./dataset-full.csv", "utf-8")
  const { data: rawData } = CSV.parse(csvData, { header: true })
  const data = rawData.map((row) => {
    const cleanedRow = {}
    for (const [key, value] of Object.entries(row)) {
      const newKey = key.trim().toLowerCase().replace(/ /g, "_").replace(/[()]/g, "")
      cleanedRow[newKey] = value
    }
    return cleanedRow
  })
  data.pop()

  // console.log(JSON.stringify(data, null, 2))

  data.forEach((row) => {
    row["gender"] = row["gender"] === "Male" ? 1 : 2
  })


  const py = await createPythonBridge()
  
  const allColumns = [...sortedHeadersRelevance, 'class']

  allColumns.forEach((column) => {
    const positiveValues = ['yes', 'positive']
    data.forEach((row) => {
      row[column] = positiveValues.includes(row[column]) ? 1 : 0
    })
  })


  const X = data.map((row) => sortedHeadersRelevance.map((col) => row[col]))
  const y = data.map((row) => row["class"])


  const { trainX, testX, trainY, testY } = train_test_split(X, y, 0.3, 42)


  const model = new GaussianNB()
  await model.init(py)

  model.fit({
    X: trainX,
    y: trainY,
  })


  const yPred = await model.predict({ X: testX })
  // console.log(yPred)




}

main().catch((err) => console.error("Error:", err)).then(() => process.exit(0))
