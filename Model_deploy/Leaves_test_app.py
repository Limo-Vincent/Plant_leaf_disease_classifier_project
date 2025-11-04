# Load libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Setup
st.set_page_config(
    page_title="Plant Leaf Disease Classifier",
    page_icon="🍃",
    layout="wide"
)

# Model & Class Name Loading


model = tf.keras.models.load_model("Model/model.keras")

# Load class names from text file
with open("Model_deploy/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the model


# Recommendations Database 
recommendations = {
    'Corn___Northern_Leaf_Blight': {
        "disease_name": "Northern Leaf Blight",
        "why": "The leaf has long, narrow tan lesions running parallel to the veins.",
        "causes": "Caused by the fungus *Exserohilum turcicum* (Setosphaeria turcica).",
        "controls": "Plant resistant hybrid seeds, apply fungicides, plant timely, and reduce previous corn residue.",
        "references": ("- [Managing Northern Corn Leaf Blight](https://www.pioneer.com/us/agronomy/Managing-Northern-Corn-Leaf-Blight.html#:~:text=Spores%20are%20produced%20on%20this,can%20result%20in%20yield%20loss.)\n"
                       "- [NCLB | University of Delaware](https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/northern-corn-leaf-blight/#:~:text=Northern%20corn%20leaf%20blight%20(NCLB)%20is%20a,quality%20in%20sweet%20corn%20and%20silage%20corn)\n"
                       "- [PlantwisePlus Knowledge Bank](https://plantwiseplusknowledgebank.org/doi/epdf/10.5555/pwkb.20117800335)")
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        "disease_name": "Gray Leaf Spot",
        "why": "Elongated necrotic spots parallel to veins with rectangular, straight-edged appearance.",
        "causes": "Caused by *Cercospora zea-maydis*.",
        "controls": "Apply fungicides, reduce previous corn residue, plant hybrid corn seeds.",
        "references": ("- [East-West Seed Plant Doctor](http://www.plantdoctor.eastwestseed.com/diagnostic-key/gray-leaf-spot?ref=dbc&plant=Corn)\n"
                       "- [Crop Protection Network](https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn)\n"
                       "- [Pioneer Seeds](https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html)")
    },
    'Corn___Common_rust': {
        "disease_name": "Common Rust",
        "why": "Brownish-red oblong pustules appearing on leaf surfaces.",
        "causes": "Caused by *Puccinia sorghi*.",
        "controls": "Use resistant hybrids, apply effective fungicides, practice crop rotation.",
        "references": ("- [Crop Protection Network](https://cropprotectionnetwork.org/encyclopedia/common-rust-of-corn)\n"
                       "- [Common Rust on Corn | UMN Extension](https://extension.umn.edu/corn-pest-management/common-rust-corn#:~:text=The%20best%20management%20practice%20is,have%20appeared%20on%20the%20leaves.)\n"
                       "- [Plantix](https://plantix.net/en/library/plant-diseases/100082/common-rust-of-maize/)")
    },
    'Corn___healthy': {
        "disease_name": "Healthy Corn",
        "why": "Leaves remain clean with no lesions or discoloration.",
        "causes": "N/A - Plant is healthy.",
        "controls": "Maintain proper agronomic practices and field monitoring.",
        "references": ("- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)\n"
                       "- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)\n"
                       "- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)")
    },
    'Potato___Early_blight': {
        "disease_name": "Potato Early Blight",
        "why": "Small brown or black spots on lower leaves surrounded by a yellow halo; lesions irregular and limited by veins.",
        "causes": "Caused by the fungus *Alternaria solani*.",
        "controls": "Crop rotation, apply registered fungicides, plant healthy seed, destroy infected material.",
        "references": ("- [Plant Health | USU Extension](https://extension.usu.edu/planthealth/ipm/notes_ag/veg-early-blight)\n"
                       "- [Agriculture Victoria](https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/target-spot-early-blight-of-potatoes#:~:text=Spores%20fall%20on%20potato%20leaves,and%20disease%20diagnosis%20and%20response.)\n"
                       "- [Greenlife Crop Protection Africa](https://www.greenlife.co.ke/early-blight/)")
    },
    'Potato___Late_blight': {
        "disease_name": "Potato Late Blight",
        "why": "Dark lesions near leaf edges and tips, rapidly expanding in cool, wet conditions.",
        "causes": "Caused by the oomycete / water mold *Phytophthora infestans*.",
        "controls": "Avoid excessive irrigation during low temperatures, Integrated Pest Management, use registered fungicides.",
        "references": ("- [Plantwise Plus Knowledge Bank](https://plantwiseplusknowledgebank.org/doi/epdf/10.1079/pwkb.20157800171)\n"
                       "- [Late Blight in Potato | NDSU Agriculture](https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato#:~:text=A%20tan%20to%20reddish%2Dbrown,Robinson%2C%20NDSU/University%20of%20Minnesota)\n"
                       "- [Plantix](https://plantix.net/en/library/plant-diseases/100040/potato-late-blight/)")
    },
    
    'Potato_healthy_aug': { 
        "disease_name": "Healthy Potato",
        "why": "Leaves are clean, green, and free from disease lesions.",
        "causes": "N/A - Plant is healthy.",
        "controls": "Proper soil drainage, crop rotation, regular inspection.",
        "references": ("- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)\n"
                       "- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)\n"
                       "- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)")
    },
    
    'Potato___healthy': { 
        "disease_name": "Healthy Potato",
        "why": "Leaves are clean, green, and free from disease lesions.",
        "causes": "N/A - Plant is healthy.",
        "controls": "Proper soil drainage, crop rotation, regular inspection.",
        "references": ("- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)\n"
                       "- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)\n"
                       "- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)")
    },
    'Tomato___Early_blight': {
        "disease_name": "Tomato Early Blight",
        "why": "Small dark spots with ring patterns that expand; leaves may yellow around lesions.",
        "causes": "Caused by the fungus *Alternaria linariae* (formerly *A. solani*).",
        "controls": "Use resistant/tolerant cultivars, apply registered fungicides such as mancozeb, maintain sufficient potassium.",
        "references": ("- [HGIC - Home & Garden Information Center](https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/)\n"
                       "- [Plantix](https://plantix.net/)\n"
                       "- [Garden Tech](https://www.gardentech.com/blog/pest-id-and-prevention/fight-blight-on-your-tomatoes#:~:text=Treating%20Blight,blight%20from%20causing%20further%20damage.)")
    },
    'Tomato___Late_blight': {
        "disease_name": "Tomato Late Blight",
        "why": "Irregular dark-brown to black water-soaked patches with pale green borders; white mold may appear on underside.",
        "causes": "Caused by the water mold pathogen *Phytophthora infestans*.",
        "controls": "Plant resilient varieties, improve ventilation and drainage, use silicate fertilizers to enhance resistance.",
        "references": ("- [HGIC - Home & Garden Information Center](https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/)\n"
                       "- [Plantix](https://plantix.net/en/library/plant-diseases/100046/tomato-late-blight/)")
    },
    'Tomato___healthy': {
        "disease_name": "Healthy Tomato",
        "why": "Leaves are uniformly green without blemishes.",
        "causes": "N/A - Plant is healthy.",
        "controls": "Proper watering, soil nutrient testing and replenishment, monitor for pests.",
        "references": ("- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)\n"
                       "- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)\n"
                       "- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)")
    },
    "default": {
        "disease_name": "Unknown",
        "why": "The model could not confidently identify this image.",
        "causes": "This may be due to poor lighting, an unclear image, or a disease not in the dataset.",
        "controls": "Please try again with a clearer, more focused image of a single leaf.",
        "references": ""
    }
}

# Sidebar Content
st.sidebar.title("🍃 About")
st.sidebar.info(
    """
    **What We Do:**
    This app uses a Deep Learning model to instantly identify 10 types of 
    plant diseases in corn, potato, and tomato leaves.
    
    **Our Mission for Kenya:**
    We built this to empower Kenyan farmers. By providing a fast, free,
    and accurate diagnosis, we help farmers detect problems early. 
    This allows for quicker treatment, reduces crop loss, and helps
    secure a healthier harvest.
    
    """
)
st.sidebar.header("How to Use")
st.sidebar.write("1. Upload a clear photo of a single leaf.")
st.sidebar.write("2. Wait for the model to analyze the image.")
st.sidebar.write("3. Get your diagnosis and treatment/prevention tips!")


# Main Page UI 
st.title("🌿 Plant Leaf Disease Classifier")
st.write(
    "Upload a corn, potato, or tomato leaf image — we'll detect if it's healthy or diseased "
    "and suggest treatment tips."
)

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    # Show example images if nothing is uploaded
    st.info("Please upload an image to get started. Or see our examples below.")
    
    st.subheader("Example Images")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(r"C:\Users\USER\Desktop\Fork\Group_8_Capstone_Project\Model_app\Images\image (612).JPG", caption="Corn (Common Rust)", width = 200)
    with col2:
        st.image(r"C:\Users\USER\Desktop\Fork\Group_8_Capstone_Project\Model_app\Images\image (58).JPG", caption="Potato (Early Blight)", width = 200)
    with col3:
        st.image(r"C:\Users\USER\Desktop\Fork\Group_8_Capstone_Project\Model_app\Images\image (156).JPG", caption="Tomato (Late Blight)", width = 200)

else:
    # Prediction Logic 
    with st.spinner('Analyzing your leaf... 🍃'):
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        img = image.resize((224, 224),Image.BILINEAR)
        img_array = np.expand_dims(np.array(img), axis=0)

        # Predict
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        confidence = 100 * np.max(preds)
        
        # This handles keys with _aug or potential mismatches.
        content = recommendations.get(pred_class, recommendations.get(pred_class.split('_aug')[0], recommendations['default']))

    st.success("Analysis complete!")
    
    
    # Display Results
    col1, col2 = st.columns([1, 2]) 

    with col1:
        st.image(image, caption="Your Uploaded Leaf", use_container_width=True)

    with col2:
        # Display prediction
        st.subheader("🧾 Diagnosis:")
        st.title(f"{content['disease_name']}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Use tabs to organize the recommendations
        tab1, tab2, tab3 = st.tabs(["About", "Treatment & Prevention", "References"])

        with tab1:
            st.subheader("Why?")
            st.write(content['why'])
            st.subheader("Causes")
            st.write(content['causes'])

        with tab2:
            st.subheader("Controls & Prevention")
            st.warning(content['controls']) 

        with tab3:
            st.subheader("References")
            st.markdown(content['references'])

# Disclaimer at the bottom
st.info(
    "**Disclaimer:** This model works best with images similar to those it was trained on. "
    "Results for images from phones or the web may vary."
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <p style='font-size:20px; font-weight:bold; color:#00FF7F;'>
            Created by: The 'Leaf It To Us' Team
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
