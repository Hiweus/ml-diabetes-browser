import { useState } from 'react'
import naiveBayesJson from './assets/model-naive.json'

import { GaussianNB } from 'ml-naivebayes'
import './index.css'

function App() {

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
    const model = new GaussianNB(true, naiveBayesJson)

    const valuesColumns = ['age', 'gender']
    const input = sortedHeadersRelevance.map((column, index) => {
      const inputElement = document.getElementById(`field-${index}`)
      if (valuesColumns.includes(column)) {
        return Number(inputElement.value)
      }
      return inputElement.checked ? 1 : 0
    })

    console.log('entrada', [input])
    const result = model.predict([input])

    console.log(result)
  }

  return (
    <div>
      <h1>Modelo de machine learning para prever diabetes</h1>
      <div className='container-input'>
        {sortedHeadersRelevance.map((column, index) => {
          return (
            <div key={column} className='input-block'>
              <label htmlFor={`field-${index}`}>{column}</label>
              {
                column === 'age' && (
                  <input type="number" id={`field-${index}`} onChange={processarResposta} defaultValue={0} />
                )
              }
              {
                column === 'gender' && (
                  <select id={`field-${index}`} onChange={processarResposta}>
                    <option value="1">Masculino</option>
                    <option value="2">Feminino</option>
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
