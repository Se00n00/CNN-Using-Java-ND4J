# **NeuralNetwork Using Java with ND4J**

Welcome to the **NeuralNetwork Using Java with ND4J** repository! This project demonstrates how to implement a Convolutional Neural Network (NeuralNetwork) in Java using the ND4J (Numerical Computing for Java) library. ND4J provides high-performance tensor computation, making it a powerful choice for machine learning and deep learning tasks in Java.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Setup Instructions](#setup-instructions)  
5. [Usage](#usage)  
6. [NeuralNetwork Architecture](#cnn-architecture)  
7. [Dataset](#dataset) 
8. [Contributing](#contributing)  

---

## **Overview**

This project is a hands-on implementation of a Convolutional Neural Network (NeuralNetwork) in Java using ND4J. CNNs are powerful neural networks widely used in computer vision tasks such as image classification, object detection, and segmentation. The repository demonstrates:

- Preprocessing data for training CNNs.
- Defining and training a NeuralNetwork using ND4J.
- Evaluating model performance on a test dataset.

This project is a great resource for Java developers interested in machine learning and deep learning.

---

## **Features**

- **Custom NeuralNetwork Implementation**: Build a NeuralNetwork model step by step in Java.
- **Flexible Architecture**: Modify layers, activation functions, and optimizers.
- **Dataset Preprocessing**: Load and normalize image data.
- **Training & Evaluation**: Train the NeuralNetwork on a sample dataset and evaluate its accuracy.

---

## **Technologies Used**

- **Java**: Core programming language.
- **ND4J**: Numerical computation library for deep learning in Java.
- **Maven**: Build automation tools.

---

## **Setup Instructions**

Follow these steps to set up and run the project:

### **Prerequisites**

1. **Java Development Kit (JDK)**  
   Ensure JDK 11 or higher is installed.  
   [Download JDK](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html)

2. **Maven**  
   Install Maven for dependency management.  
   [Maven Installation Guide](https://maven.apache.org/install.html)

3. **Clone the Repository**  
   ```bash
   git clone https://github.com/Se00n00/NeuralNetwork-Using-Java-ND4J.git
   cd NeuralNetwork-Using-Java-ND4J
## **Usage**

### **Step 1: Get all the dependecies**
- Reload the pom.xml to resolve all the required dependencies
- In case of `artifact nor found error`, visit maven repository to get the info of correct version

### **Step 2: Clear Resources and Mention your Dataset Directory**
- If You wish to work on your own dataset, clear the resources first (any thing iniside it)
- Inside `Main.java` change the already mention path of your dataset to your dataset in local

### **Step 3: Train the Model**
- Run the `Main.java` file to start the training process:
  ```bash
  mvn exec:java -Dexec.mainClass="com.example.Main"
- It will first check if the preprocessed data exists in resources or not if not, then would prepare the dataset
- Then the actual training of model would start

## **NeuralNetwork Architecture**

### **Example Architecture Definition : METHOD : 1**
  ```java
  NeuralNetwork NN = new NeuralNetwork()
  // Add Layer To It
  NN.add(new Conv2D(10,7,0,4)0;        // (Number of Filters, FIlter Shape, Padding, Strides )
  NN.add(new MaxPool2D(2,1));          // (Window Shape, Strides)
  NN.add(new Flatten());               // Flatten Defore Dense Layer
  NN.add(new Dense(512, "RELU"));      // (Number of Neurons, Activation Function
  NN.add(new Dense(10, "SOFTMAX"));
  ```
### ** Example Architecture Definition :  METHOD : 2**
```java
NeuralNetwork NN = new NeuralNetwork(new ArrayList<>(Arrays.asList(
                new Conv2D(10,7,0,4),
                new MaxPool2D(2,1),
                new Flatten(),
                new Dense(512, "RELU"),
                new Dense(64, "RELU"),
                new Dense(10,"SOFTMAX")
 )));
```
### ** Training And Evaluation**
```java
// Train the Model
NN.fit(Train_X, Train_Y, 0.001,10);    // (Training_X, Training_Y, Learning Rate, Epoch)

// Evalute the Model
System.out.println("Final Accuracy :: "+NN.Accuracy(Test_X, Test_Y)+"%");
```
Modify the architecture in `Main.java` to suit your requirements Using any Method.

---

## **Contributing**

Contributions are welcome! If you'd like to contribute to this project, follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right of this page.
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/Se00n00/NeuralNetwork-Using-Java-ND4J.git
