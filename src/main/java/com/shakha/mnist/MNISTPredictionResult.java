package com.shakha.mnist;


import lombok.*;

@Setter
@Getter
@AllArgsConstructor
public class MNISTPredictionResult {
    private int predictedDigit;
    private float[] probabilities;
}