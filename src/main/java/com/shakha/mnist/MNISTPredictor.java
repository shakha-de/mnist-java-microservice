package com.shakha.mnist;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import java.util.Arrays;

public class MNISTPredictor {
    public static void main(String[] args) {
        String modelPath = "saved_model";

        // Die Eingabe muss 3D sein, um zur Modell-Signatur zu passen.
        float[][][] inputData = new float[1][28][28];

        // Dummy pattern
        for (int i = 0; i < 28; i++) {
            inputData[0][i][i] = 1.0f;
        }
        System.out.println("Created a dummy 28x28 input tensor.");

        try (
                SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
                Session session = model.session()
        ) {
            System.out.println("TensorFlow model loaded successfully from: " + modelPath);

            try (Tensor inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(inputData))) {

                System.out.println("Running model prediction...");


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

                System.out.println("--------------------");
                System.out.println("Model Prediction Result:");
                System.out.println("Predicted Digit: " + predictedDigit);
                System.out.println("Probabilities: " + Arrays.toString(probabilities));
                System.out.println("--------------------");
            }
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
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