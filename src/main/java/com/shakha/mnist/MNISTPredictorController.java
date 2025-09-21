package com.shakha.mnist;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/mnist")
public class MNISTPredictorController {

    private final MNISTPredictor mnistPredictor;

    @PostMapping("/predict")
    public ResponseEntity<MNISTPredictionResult> predict(@RequestBody float[][][] input) {
        try {
            MNISTPredictionResult result = mnistPredictor.predict(input);
            return ResponseEntity.ok(result);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}