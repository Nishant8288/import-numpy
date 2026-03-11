import numpy as np

sensor_vector = np.array([25, 101.3, 60])
print("Sensor Vector:")
print(sensor_vector)

# Sensor data matrix (multiple machine readings)
sensor_matrix = np.array([
    [25, 101.3, 60],
    [26, 100.9, 62],
    [24, 101.5, 59],
    [27, 101.0, 63]
])

print("\nSensor Data Matrix:")
print(sensor_matrix)

# Calculate mean values of sensors
mean_values = np.mean(sensor_matrix, axis=0)

print("\nAverage Sensor Values:")
print(mean_values)

# Weight vector used in ML model
weights = np.array([0.5, 0.3, 0.2])

# Matrix multiplication for prediction
prediction = np.dot(sensor_matrix, weights)

print("\nPredicted Output:")
print(prediction)
