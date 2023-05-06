const modelURL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json'
    let model = undefined

    async function loadModel() {
        const model = await tf.loadLayersModel(modelURL)
        model.summary()

        //   batch of 1
        const input = tf.tensor2d([[700]])

        // batch of 3
        const inputBatch = tf.tensor2d([[1100], [8000], [700]])

        // batch of 5
        const inputBatch5 = tf.tensor2d([[1100], [8000], [700], [400], [2000]])

        const result = model.predict(input)
        const resultBatch = model.predict(inputBatch)
        const resultBatch5 = model.predict(inputBatch5)

        result.print()
        resultBatch.print()
        resultBatch5.print()

        input.dispose()
        inputBatch.dispose()
        inputBatch5.dispose()
        result.dispose()
        resultBatch.dispose()
        resultBatch5.dispose()
        model.dispose()

    }

    loadModel()
