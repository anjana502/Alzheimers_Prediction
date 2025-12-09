This is a multi-class classification problem where the model must distinguish between four stages of Alzheimer’s disease severity, based on the Clinical Dementia Rating (CDR) scale: 

* Non-Demented (Non) 
* Very Mild Demented (Very Mild) 
* Mild Demented (Mild)
* Moderate Demented (Moderate)

The system architecture includes the following components: 

* MRI Scans data:
    * CNN: A foundational TensorFlow-based CNN model is used to predict the patient’s dementia stage from their MRI scan images without any additional data pre-processing steps.
    * Data augmentation: The class imbalance in the data samples is reduced using data augmentation to generate new data samples by performing transformations such as zooming in, out, rotating, etc. on the existing images.
    * Grad CAM: As CNN are generally considered as “black boxes” as they do not provide any information about how they arrived at a specific output, the Grad CAM method is used to highlight regions of the MRI scan which led to model making that prediction. Grad CAM provides a heatmap to overlay the input image by extracting the gradients flowing back from the last convolutional layer in the model.
    * SHAP: SHAP is used to obtain feature importance to understand each feature’s contribution to the predictions of the foundational and data augmented CNN models.
    * Ensemble methods: Ensemble methods such as DecisionTree, Bagging, and RandomForest are used to predict the patient’s dementia stage based on the pre-processed image bytes data by performing K-fold cross-validation to obtain better accuracy.
    * KNN: A KNN classification model with K = 5 (no. of nearest neighbors) is used to predict the patients’ dementia stage.
    * SVM: A SVM model is trained using RBF and Polynomial kernels to classify the MRI scan images. The images are pre-processed to extract the important features using the VGG16 model.
* Handwriting data: Patient’s handwriting data has a unique relationship with the presence of Alzheimer’s disease. This data is used to predict the presence of Alzheimer’s in patients by performing PCA to reduce dimensionality of the data and training an XGBoost model to classify whether a patient has Alzheimer’s or not by analyzing the patterns in the obtained principal components.
