# Multi-stage build for Spring Boot app with TensorFlow Java

FROM maven:3-openjdk-17 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -B
COPY src ./src
RUN mvn clean package -DskipTests

# Runtime stage
FROM openjdk:17-slim
WORKDIR /app

# Install TensorFlow C library (required for TensorFlow Java)
RUN apt-get update && apt-get install -y wget && \
    wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.13.0.tar.gz && \
    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.13.0.tar.gz && \
    ldconfig && \
    rm libtensorflow-cpu-linux-x86_64-2.13.0.tar.gz && \
    apt-get remove -y wget && apt-get autoremove -y && apt-get clean

# Copy the built JAR
COPY --from=build /app/target/*.jar app.jar

# Copy the saved model directory
COPY saved_model ./saved_model

# Expose port 8080 (default Spring Boot port)
EXPOSE 8080

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]