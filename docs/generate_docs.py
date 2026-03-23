import os
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

docs_dir = os.path.dirname(os.path.abspath(__file__))

def create_architecture_diagram():
    img = Image.new('RGB', (800, 650), color=(13, 17, 23))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 26)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    draw.text((300, 20), "FraudGuard Architecture", fill=(88, 166, 255), font=title_font)
    
    flow = [
        "[ Raw CSV Upload / sample.csv ]",
        "         ↓",
        "[ cleaner.py: Remove dupes, Normalize amounts / strings ]",
        "         ↓",
        "[ features.py: Feature Eng (velocity, city, flags) ]",
        "         ↓",
        "[ model.py: RF Classifier trained on Pseudo-labels ]",
        "         ↓",
        "[ analyzer.py: Metric compilation, aggregations ]",
        "         ↓",
        "[ app.py: Flask JSON Response via POST ]",
        "         ↓",
        "[ index.html : DOM manipulations with Chart.js ]"
    ]
    
    y = 80
    for line in flow:
        draw.text((150, y), line, fill=(201, 209, 217), font=font)
        y += 45
        
    img.save(os.path.join(docs_dir, "architecture_diagram.png"))

def create_technical_summary():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Technical Summary - FraudGuard Dashboard", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    
    content = [
        "1. Overview",
        "Provides 6 deliverables using a Python analytics backend (Flask/scikit-learn)",
        "integrated with a stateless singular webpage frontend.",
        "",
        "2. Data Processing Pipeline",
        "Pipeline identifies structural disparities: duplicate rows, missing amounts,",
        "or invalid subnets dynamically parsing fields before running ML extraction.",
        "",
        "3. ML Architecture",
        "A Random Forest algorithm determines probabilities through pseudo-labels",
        "aggregated from feature thresholds evaluated independently over variables.",
        "",
        "4. Fraud Patterns Highlighted",
        "Pattern 1 - Late Night + High Amount + Cross-City",
        "Transactions 00:00-05:00 with huge amounts transacting across differing regions.",
        "",
        "Pattern 2 - Invalid IP + New Device + Card Payment",
        "Malformed origin IP metrics originating from an unrecognized hardware device.",
        "",
        "Pattern 3 - High Velocity + High Amount-to-Balance Ratio",
        ">10 account events inside a rolling 7-day period exhausting balance buffers.",
        "",
        "5. Deployment strategy",
        "Ready to spin up without external components on Vercel/Render.",
        "",
        "6. Frontend",
        "Implemented using TailwindCSS providing native dark-mode visual elements."
    ]
    
    for line in content:
        pdf.cell(200, 8, txt=line, ln=True)
        
    pdf.output(os.path.join(docs_dir, "technical_summary.pdf"))

if __name__ == "__main__":
    create_architecture_diagram()
    create_technical_summary()
    print("Documents generated successfully in", docs_dir)
