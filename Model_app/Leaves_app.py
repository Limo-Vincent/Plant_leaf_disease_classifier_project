
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load class names from text file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
model = tf.keras.models.load_model("model.keras")

# Disease recommendations dictionary
recommendations = {
    'Corn___Northern_Leaf_Blight': """
### Northern Leaf Blight

**Why:** The leaf has long, narrow tan lesions running parallel to the veins.

**Causes:** Caused by the fungus *Exserohilum turcicum* (Setosphaeria turcica).

**Controls:** Plant resistant hybrid seeds, apply fungicides, plant timely, and reduce previous corn residue.

**References:**
- [Managing Northern Corn Leaf Blight](https://www.pioneer.com/us/agronomy/Managing-Northern-Corn-Leaf-Blight.html#:~:text=Spores%20are%20produced%20on%20this,can%20result%20in%20yield%20loss.)
- [NCLB | University of Delaware](https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/northern-corn-leaf-blight/#:~:text=Northern%20corn%20leaf%20blight%20(NCLB)%20is%20a,quality%20in%20sweet%20corn%20and%20silage%20corn)
- [PlantwisePlus Knowledge Bank](https://plantwiseplusknowledgebank.org/doi/epdf/10.5555/pwkb.20117800335)
""",

    'Corn___Cercospora_leaf_spot Gray_leaf_spot': """
### Gray Leaf Spot

**Why:** Elongated necrotic spots parallel to veins with rectangular, straight-edged appearance.

**Causes:** Caused by *Cercospora zea-maydis*.

**Controls:** Apply fungicides, reduce previous corn residue, plant hybrid corn seeds.

**References:**
- [East-West Seed Plant Doctor](http://www.plantdoctor.eastwestseed.com/diagnostic-key/gray-leaf-spot?ref=dbc&plant=Corn)
- [Crop Protection Network](https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn)
- [Pioneer Seeds](https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html)
""",

    'Corn___Common_rust': """
### Common Rust

**Why:** Brownish-red oblong pustules appearing on leaf surfaces.

**Causes:** Caused by *Puccinia sorghi*.

**Controls:** Use resistant hybrids, apply effective fungicides, practice crop rotation.

**References:**
- [Crop Protection Network](https://cropprotectionnetwork.org/encyclopedia/common-rust-of-corn)
- [Common Rust on Corn | UMN Extension](https://extension.umn.edu/corn-pest-management/common-rust-corn#:~:text=The%20best%20management%20practice%20is,have%20appeared%20on%20the%20leaves.)
- [Plantix](https://plantix.net/en/library/plant-diseases/100082/common-rust-of-maize/)
""",

    'Corn___healthy': """
### Healthy Corn

**Why:** Leaves remain clean with no lesions or discoloration.

**Controls:** Maintain proper agronomic practices and field monitoring.

**References:**
- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)
- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)
- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)
""",

    'Potato___Early_blight': """
### Potato Early Blight

**Why:** Small brown or black spots on lower leaves surrounded by a yellow halo; lesions irregular and limited by veins.

**Causes:** Caused by the fungus *Alternaria solani*.

**Controls:** Crop rotation, apply registered fungicides, plant healthy seed, destroy infected material.

**References:**
- [Plant Health | USU Extension](https://extension.usu.edu/planthealth/ipm/notes_ag/veg-early-blight)
- [Agriculture Victoria](https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/target-spot-early-blight-of-potatoes#:~:text=Spores%20fall%20on%20potato%20leaves,and%20disease%20diagnosis%20and%20response.)
- [Greenlife Crop Protection Africa](https://www.greenlife.co.ke/early-blight/)
""",

    'Potato___Late_blight': """
### Potato Late Blight

**Why:** Dark lesions near leaf edges and tips, rapidly expanding in cool, wet conditions.

**Causes:** Caused by the oomycete / water mold *Phytophthora infestans*.

**Controls:** Avoid excessive irrigation during low temperatures, Integrated Pest Management, use registered fungicides.

**References:**
- [Plantwise Plus Knowledge Bank](https://plantwiseplusknowledgebank.org/doi/epdf/10.1079/pwkb.20157800171)
- [Late Blight in Potato | NDSU Agriculture](https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato#:~:text=A%20tan%20to%20reddish%2Dbrown,Robinson%2C%20NDSU/University%20of%20Minnesota)
- [Plantix](https://plantix.net/en/library/plant-diseases/100040/potato-late-blight/)
""",

    'Potato___healthy': """
### Healthy Potato

**Why:** Leaves are clean, green, and free from disease lesions.

**Controls:** Proper soil drainage, crop rotation, regular inspection.

**References:**
- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)
- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)
- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)
""",

    'Tomato___Early_blight': """
### Tomato Early Blight

**Why:** Small dark spots with ring patterns that expand; leaves may yellow around lesions.

**Causes:** Caused by the fungus *Alternaria linariae* (formerly *A. solani*).

**Controls:** Use resistant/tolerant cultivars, apply registered fungicides such as mancozeb, maintain sufficient potassium.

**References:**
- [HGIC - Home & Garden Information Center](https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/)
- [Plantix](https://plantix.net/)
- [Garden Tech](https://www.gardentech.com/blog/pest-id-and-prevention/fight-blight-on-your-tomatoes#:~:text=Treating%20Blight,blight%20from%20causing%20further%20damage.)
""",

    'Tomato___Late_blight': """
### Tomato Late Blight

**Why:** Irregular dark-brown to black water-soaked patches with pale green borders; white mold may appear on underside.

**Causes:** Caused by the water mold pathogen *Phytophthora infestans*.

**Controls:** Plant resilient varieties, improve ventilation and drainage, use silicate fertilizers to enhance resistance.

**References:**
- [HGIC - Home & Garden Information Center](https://hgic.clemson.edu/factsheet/tomato-diseases-disorders/)
- [Plantix](https://plantix.net/en/library/plant-diseases/100046/tomato-late-blight/)
""",

    'Tomato___healthy': """
### Healthy Tomato

**Why:** Leaves are uniformly green without blemishes.

**Controls:** Proper watering, soil nutrient testing and replenishment, monitor for pests.

**References:**
- [Elite Tree Care](https://www.elitetreecare.com/2021/02/maintaining-plant-health-tips/#:~:text=Clean%20out%20the%20garden%20in,morning%2C%20especially%20on%20hot%20days.)
- [Koppert Kenya](https://www.koppert.co.ke/disease-control/)
- [Cropler](https://www.cropler.io/blog-posts/crop-diseases)
"""
}

# Streamlit UI
st.title("🌿 Plant Disease Classifier with Recommendations")
st.write("Upload a corn, potato, or tomato leaf image — we'll detect if it's healthy or diseased and suggest treatment tips.")

# Add space
st.write("")
st.write("") 

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file:
        # Load and show image
        image = Image.open(uploaded_file).convert("RGB")
        # Create sub-columns inside col2 to push the image slightly right
        col_left, col_image = st.columns([0.5, 2])  
        with col_image:
            st.image(image, caption="Uploaded Image", width = 200)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)

        # Predict
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]

        # Remove _aug suffix if present
        pred_class_clean = pred_class.split('_aug')[0]
        confidence = 100 * np.max(preds)

        # Display prediction
        st.subheader(f"🧾 Prediction: {pred_class_clean}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Show recommendations
        if pred_class_clean in recommendations:
            st.markdown(recommendations[pred_class_clean], unsafe_allow_html=True)
        else:
            st.warning("No recommendation available for this class.")

# Disclaimer at the bottom
st.info(
    "**Disclaimer:** This model works best with images similar to those it was trained on. "
    "Results for images from phones or the web may vary."
)
