import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions) # Return index of max element and its value

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    try:
        if(st.button("Show Image")):
            st.image(test_image,width=4,use_column_width=True)
    except:
        st.write("Please upload an image.")

    try:
        #Predict button
        if(st.button("Predict")):
            result_index, confidence = model_prediction(test_image)
            st.snow()
            st.write("Our Prediction")
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
            st.success(f"Model is predicting it's a {class_name[result_index]} with {confidence * 100:.2f}% confidence")

            # Disease Causes and Treatments
            disease_info = {
                0: ("Apple Scab", "Caused by the fungus Venturia inaequalis. Can be treated with fungicides and by removing infected leaves."),
                1: ("Apple Black Rot", "Caused by the fungus Botryosphaeria obtusa. Can be treated by pruning and removing infected plant parts."),
                2: ("Apple Cedar Rust", "Caused by the fungus Gymnosporangium juniperi-virginianae. Can be treated with fungicides and by removing nearby juniper trees."),
                3: ("Healthy Apple Leaf", "No disease detected. The plant is healthy."),
                4: ("Healthy Blueberry", "No disease detected. The plant is healthy."),
                5: ("Cherry Powdery Mildew", "Caused by the fungus Podosphaera clandestina. Can be treated with fungicides and by improving air circulation."),
                6: ("Healthy Cherry Leaf", "No disease detected. The plant is healthy."),
                7: ("Corn Cercospora Leaf Spot", "Caused by the fungus Cercospora zeae-maydis. Can be treated with fungicides and crop rotation."),
                8: ("Corn Common Rust", "Caused by the fungus Puccinia sorghi. Can be treated with fungicides and resistant hybrids."),
                9: ("Corn Northern Leaf Blight", "Caused by the fungus Exserohilum turcicum. Can be treated with fungicides and resistant hybrids."),
                10: ("Healthy Corn Leaf", "No disease detected. The plant is healthy."),
                11: ("Grape Black Rot", "Caused by the fungus Guignardia bidwellii. Can be treated with fungicides and by removing infected plant parts."),
                12: ("Grape Esca (Black Measles)", "Caused by a complex of fungi. Can be treated with fungicides and by removing infected plant parts."),
                13: ("Grape Leaf Blight", "Caused by the fungus Pseudopeziza tracheiphila. Can be treated with fungicides and by removing infected plant parts."),
                14: ("Healthy Grape Leaf", "No disease detected. The plant is healthy."),
                15: ("Orange Citrus Greening", "Caused by the bacteria Candidatus Liberibacter asiaticus. Can be treated by removing infected trees and controlling the insect vector."),
                16: ("Peach Bacterial Spot", "Caused by the bacterium Xanthomonas campestris. Can be treated with bactericides and by removing infected plant parts."),
                17: ("Healthy Peach Leaf", "No disease detected. The plant is healthy."),
                18: ("Pepper Bacterial Spot", "Caused by the bacterium Xanthomonas campestris. Can be treated with bactericides and by removing infected plant parts."),
                19: ("Healthy Pepper Leaf", "No disease detected. The plant is healthy."),
                20: ("Potato Early Blight", "Caused by the fungus Alternaria solani. Can be treated with fungicides and crop rotation."),
                21: ("Potato Late Blight", "Caused by the oomycete Phytophthora infestans. Can be treated with fungicides and resistant varieties."),
                22: ("Healthy Potato Leaf", "No disease detected. The plant is healthy."),
                23: ("Healthy Raspberry", "No disease detected. The plant is healthy."),
                24: ("Healthy Soybean", "No disease detected. The plant is healthy."),
                25: ("Squash Powdery Mildew", "Caused by various fungi. Can be treated with fungicides and by improving air circulation."),
                26: ("Strawberry Leaf Scorch", "Caused by the fungus Diplocarpon earlianum. Can be treated with fungicides and by removing infected plant parts."),
                27: ("Healthy Strawberry Leaf", "No disease detected. The plant is healthy."),
                28: ("Tomato Bacterial Spot", "Caused by the bacterium Xanthomonas campestris. Can be treated with bactericides and by removing infected plant parts."),
                29: ("Tomato Early Blight", "Caused by the fungus Alternaria solani. Can be treated with fungicides and crop rotation."),
                30: ("Tomato Late Blight", "Caused by the oomycete Phytophthora infestans. Can be treated with fungicides and resistant varieties."),
                31: ("Tomato Leaf Mold", "Caused by the fungus Passalora fulva. Can be treated with fungicides and by improving air circulation."),
                32: ("Tomato Septoria Leaf Spot", "Caused by the fungus Septoria lycopersici. Can be treated with fungicides and by removing infected plant parts."),
                33: ("Tomato Spider Mites", "Caused by the mite Tetranychus urticae. Can be treated with miticides and by maintaining plant health."),
                34: ("Tomato Target Spot", "Caused by the fungus Corynespora cassiicola. Can be treated with fungicides and by removing infected plant parts."),
                35: ("Tomato Yellow Leaf Curl Virus", "Caused by a virus transmitted by whiteflies. Can be treated by controlling the whitefly population."),
                36: ("Tomato Mosaic Virus", "Caused by the tomato mosaic virus. Can be treated by removing infected plants and avoiding contaminated tools."),
                37: ("Healthy Tomato Leaf", "No disease detected. The plant is healthy.")
            }
        
            if result_index in disease_info:
                disease, treatment = disease_info[result_index]
                st.write(f"**Disease:** {disease}")
                st.write(f"**Cause:** {disease_info[result_index][1].split('.')[0]}")
                st.write(f"**Treatment:** {disease_info[result_index][1].split('.')[1]}")
            else:
                st.write("Unknown case")
    except:
        st.write("Please upload an image.")