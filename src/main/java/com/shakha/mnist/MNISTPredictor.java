package com.shakha.mnist;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;


@Service
@Slf4j
public class MNISTPredictor {

    private SavedModelBundle model;
    private Session session;

    @PostConstruct
    public void init() {
        String modelPath = "saved_model";
        try {
            model = SavedModelBundle.load(modelPath, "serve");
            session = model.session();
            log.info("TensorFlow model loaded successfully from: {} ", modelPath);
        } catch (Exception e) {
            log.error("Failed to load TensorFlow model: {} ", e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @PreDestroy
    public void cleanup() {
        if (session != null) {
            session.close();
        }
        if (model != null) {
            model.close();
        }
    }

    public MNISTPredictionResult predict(float[][][] inputData) {
        if (inputData.length != 1 || inputData[0].length != 28 || inputData[0][0].length != 28) {
            throw new IllegalArgumentException("Input must be a 3D array of shape [1][28][28]");
        }

        try (Tensor inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(inputData))) {
            Tensor result = session.runner()
                    .feed("serving_default_keras_tensor_15", inputTensor)
                    .fetch("StatefulPartitionedCall_1")
                    .run()
                    .get(0);

            float[][] outputMatrix = new float[1][10];
            try (TFloat32 resultTensor = (TFloat32) result) {
                StdArrays.copyFrom(resultTensor, outputMatrix);
            }

            float[] probabilities = outputMatrix[0];
            int predictedDigit = findMaxIndex(probabilities);

            return new MNISTPredictionResult(predictedDigit, probabilities);
        } catch (Exception e) {
            throw new RuntimeException("Prediction failed: " + e.getMessage(), e);
        }
    }

    private static int findMaxIndex(float[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}