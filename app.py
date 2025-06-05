import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
import timm

# Disease info
disease_info = {
    "cataract": {
        "causes": [
            "Aging (most common cause)",
            "Diabetes",
            "Eye trauma or injury",
            "Prolonged steroid use",
            "Excessive UV light exposure"
        ],
        "treatment": [
            "Early stages: Stronger glasses/lighting",
            "Advanced cases: Surgery (phacoemulsification)",
            "Intraocular lens implantation"
        ],
        "prevention": [
            "Wear UV-protective sunglasses outdoors",
            "Manage diabetes carefully",
            "Quit smoking",
            "Eat antioxidant-rich foods (leafy greens, fruits)",
            "Get regular eye exams after age 40"
        ]
    },
    "diabetic_retinopathy": {
        "causes": [
            "Long-term uncontrolled diabetes",
            "High blood sugar damaging retinal blood vessels",
            "High blood pressure",
            "High cholesterol"
        ],
        "treatment": [
            "Blood sugar control (primary treatment)",
            "Anti-VEGF injections (Lucentis, Eylea)",
            "Laser treatment (photocoagulation)",
            "Vitrectomy in advanced cases"
        ],
        "prevention": [
            "Maintain HbA1c below 7%",
            "Control blood pressure (<130/80 mmHg)",
            "Annual dilated eye exams if diabetic",
            "Quit smoking",
            "Regular physical activity"
        ]
    },
    "glaucoma": {
        "causes": [
            "High intraocular pressure",
            "Family history",
            "Age over 60",
            "Certain medical conditions",
            "Long-term steroid use"
        ],
        "treatment": [
            "Prescription eye drops",
            "Oral medications",
            "Laser therapy",
            "Surgery (trabeculectomy)"
        ],
        "prevention": [
            "Regular eye pressure checks after age 40",
            "Exercise moderately (avoid head-down positions)",
            "Protect eyes from injury",
            "Limit caffeine intake",
            "Elevate head while sleeping if at risk"
        ]
    },
    "normal": {
        "causes": ["Healthy eye tissue"],
        "treatment": ["No treatment needed"],
        "prevention": [
            "Annual comprehensive eye exams",
            "20-20-20 rule (every 20 mins, look 20 feet away for 20 sec)",
            "Wear protective eyewear during sports",
            "Maintain a balanced diet rich in omega-3s"
        ]
    }
}

# Load model
model = timm.create_model('resnet18', pretrained=False, num_classes=4)
model.load_state_dict(torch.load("eye_model.pth", map_location=torch.device('cpu')))
model.eval()

# Class names
class_names = list(disease_info.keys())

# Formatter
def format_section(title, items):
    return f"\n**{title}**\n" + "\n".join([f"• {item}" for item in items])

# Predictor
def predict_eye_disease(image):
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = image.convert("RGB")
    img_tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_class = class_names[probs.argmax()]
    confidence = probs.max().item()
    info = disease_info[predicted_class]

    result = [
        f"## 🔍 Prediction: **{predicted_class.upper()}** ({confidence:.1%} confidence)",
        format_section("🩺 Possible Causes", info["causes"]),
        format_section("💊 Recommended Treatments", info["treatment"]),
        format_section("🛡️ Prevention Tips", info["prevention"]),
        "\n⚠️ **Important**: This AI assessment cannot replace professional medical diagnosis. Please consult an ophthalmologist."
    ]

    return "\n".join(result)

# UI
css = """
#diagnosis-box {
    font-family: Arial, sans-serif;
    line-height: 1.6;
}
.markdown-text {
    white-space: pre-line;
}
"""

with gr.Blocks(css=css) as app:
    gr.Markdown("# 👁️ AI Eye Disease Detector")
    gr.Markdown("Upload or capture an eye image for disease assessment")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Eye Image", sources=["upload", "webcam"])
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            diagnosis_output = gr.Markdown(
                label="Diagnosis Report",
                elem_id="diagnosis-box",
                elem_classes="markdown-text"
            )

    submit_btn.click(
        fn=predict_eye_disease,
        inputs=image_input,
        outputs=diagnosis_output
    )

app.launch(server_name="0.0.0.0", server_port=8080)
