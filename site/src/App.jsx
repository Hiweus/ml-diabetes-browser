import { useState } from 'react'
import naiveBayesJson from './assets/model-naive.json'
import treeJson from './assets/model-tree.json'
import forestJson from './assets/model-random.json'

import { GaussianNB } from 'ml-naivebayes'
import { DecisionTreeClassifier } from 'ml-cart'
import { RandomForestClassifier } from 'ml-random-forest'
import './index.css'

function App() {

  const [prevision, setPrevision] = useState({
    naive: 0,
    tree: 0,
    forest: 0,
  })

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
  function processarResposta() {
    const naiveModel = new GaussianNB(true, naiveBayesJson)
    const treeModel = new DecisionTreeClassifier(true, treeJson)
    const forestModel = new RandomForestClassifier(true, forestJson)

    const valuesColumns = ['age', 'gender']
    const input = sortedHeadersRelevance.map((column, index) => {
      const inputElement = document.getElementById(`field-${index}`)
      if (valuesColumns.includes(column)) {
        return Number(inputElement.value)
      }
      return inputElement.checked ? 1 : 0
    })

    const resultNaive = naiveModel.predict([input])
    const resultTree = treeModel.predict([input])
    const resultForest = forestModel.predict([input])

    const result = {
      naive: resultNaive.at(0),
      tree: resultTree.at(0),
      forest: resultForest.at(0),
    }
    setPrevision(result)

    console.log({
      result,
      input,
    })
  }

  return (
    <div>
      <h1>Modelo de machine learning para prever diabetes</h1>


      <div style={{ display: 'flex', flexDirection: 'column', fontSize: '20px' }}>
        <div><b>Random forest</b> <span className={prevision.forest ? 'danger': ''}>{prevision.forest ? 'SIM' : 'NÃO'}</span></div>
        <div><b>Decision tree</b> <span className={prevision.tree ? 'danger': ''}>{prevision.tree ? 'SIM' : 'NÃO'}</span></div>
        <div><b>Naive bayes</b> <span className={prevision.naive ? 'danger': ''}>{prevision.naive ? 'SIM' : 'NÃO'}</span></div>
      </div>

      <div className='container-input'>
        {sortedHeadersRelevance.map((column, index) => {
          return (
            <div key={column} className='input-block'>
              <label htmlFor={`field-${index}`}>{column}</label>
              {
                column === 'age' && (
                  <input type="number" id={`field-${index}`} onChange={processarResposta} defaultValue={67} />
                )
              }
              {
                column === 'gender' && (
                  <select id={`field-${index}`} onChange={processarResposta}>
                    <option value="0">Masculino</option>
                    <option value="1">Feminino</option>
                  </select>
                )
              }


              {
                !['gender', 'age'].includes(column) && (
                  <input type="checkbox" id={`field-${index}`} onChange={processarResposta} />
                )
              }
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default App
