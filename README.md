<h1>Skin Cancer Detection Using CNN</h1>
        <h2>Project Overview</h2>
        <p>The objective of this project is to develop a Convolutional Neural Network (CNN) model that can detect melanoma, a type of skin cancer, using dermatoscopic images. The model aims to classify images into 'Benign' (non-cancerous) or 'Malignant' (cancerous) with an accuracy of approximately 90%.</p>      
        <h2>Dataset</h2>
        <p>The project utilizes the HAM10000 dataset, which comprises 10,000 dermatoscopic images labeled as 'Benign' or 'Malignant'. This dataset is a comprehensive collection of skin lesion images that are crucial for training and evaluating the CNN model.</p>
        <p><a href="https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000" target="_blank">HAM10000 Dataset on Kaggle</a></p>
                <h2>Technologies Used</h2>
        <ul>
            <li><strong>TensorFlow:</strong> An open-source machine learning framework for building and training neural networks.</li>
            <li><strong>Keras:</strong> A high-level API for building and training deep learning models, integrated into TensorFlow.</li>
            <li><strong>Streamlit:</strong> An open-source app framework for creating and sharing custom web applications for machine learning and data science.</li>
        </ul>
            <h2>Model Architecture</h2>
        <p>The CNN model architecture includes the following components:</p>
        <ul>
            <li><strong>Convolutional Layers:</strong> These layers perform convolution operations to detect various features in the input images. Each convolutional layer is followed by an activation function (typically ReLU) to introduce non-linearity.</li>
            <li><strong>Max-Pooling Layers:</strong> These layers perform down-sampling operations to reduce the spatial dimensions of the feature maps while retaining the most important information.</li>
            <li><strong>Fully Connected Layers:</strong> After feature extraction, the output of the convolutional and pooling layers is flattened and fed into fully connected layers. These layers perform the final classification based on the extracted features.</li>
            <li><strong>Output Layer:</strong> The final fully connected layer outputs the probability of the input image being 'Benign' or 'Malignant' using a softmax activation function.</li>
        </ul>       
        <h2>Data Augmentation</h2>
        <p>To improve the generalization of the model and prevent overfitting, various data augmentation techniques are applied to the training images. These techniques include:</p>
        <ul>
            <li>Rotation</li>
            <li>Zoom</li>
            <li>Horizontal and vertical flips</li>
            <li>Shifts (width and height)</li>
        </ul>    
        <h2>Model Performance</h2>
        <p>The model achieved an accuracy of approximately 90% on the validation set. Below is the output for accuracy and loss during training:</p>
        <img src="">
        <pre aling="center">
        <p><strong>Loss:</strong> 0.2284</p>
        <p><strong>Accuracy:</strong> 0.9000</p>
        </pre>
        <h3>Confusion Matrix</h3>
        <img src="path_to_confusion_matrix.png" alt="Confusion Matrix">
        <h2>Deployment</h2>
        <p>The trained model is deployed using Streamlit to create a user-friendly web application for skin cancer detection. The application allows users to upload an image and receive a prediction on whether the lesion is benign or malignant.</p>
        <h2>Instructions to Run the Project</h2>
        <ol>
            <li><strong>Clone the repository:</strong></li>
            <pre>
<code>
git clone https://github.com/safwannasir49/skin-cancer-detection.git
cd skin-cancer-detection
</code>
        </ol>
        <h2>Conclusion</h2>
        <p>This project demonstrates the application of Convolutional Neural Networks in detecting melanoma from dermatoscopic images. By leveraging the HAM10000 dataset and employing data augmentation techniques, the model achieves high accuracy and can be effectively used as a tool for skin cancer detection.</p>
        
